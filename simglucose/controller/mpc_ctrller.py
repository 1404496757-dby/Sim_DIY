import numpy as np
from cvxpy import *
from .base import Controller, Action
from simglucose.simulation.env import T1DSimEnv
from simglucose.patient.t1dpatient import T1DPatient


class MPController(Controller):
    def __init__(self, patient_name='adult#001', prediction_horizon=6, control_horizon=3):
        """
        MPC控制器初始化
        Args:
            patient_name: 患者ID，用于获取模型参数
            prediction_horizon: 预测时域(30分钟为单位)
            control_horizon: 控制时域(30分钟为单位)
        """
        self.patient = T1DPatient.withName(patient_name)
        self.N = prediction_horizon
        self.M = control_horizon

        # 目标血糖范围 (mg/dL)
        self.target = 120
        self.safe_min = 70
        self.safe_max = 180

        # 控制参数
        # self.basal_max = 3 * self.patient._basal  # 最大基础率
        self.basal_max = 35  # 最大基础率
        self.bolus_max = 5  # 最大bolus剂量(U)

        # 权重矩阵
        self.Q = 1.0  # 血糖误差权重
        self.R_basal = 0.1  # basal变化权重
        self.R_bolus = 0.5  # bolus剂量权重

        # 状态空间模型 (简化UVa/Padova模型)
        self._init_model()

        # 状态估计
        self.x_est = np.zeros(3)  # [葡萄糖, 胰岛素, 碳水化合物]
        self.last_CGM = None

    def _init_model(self):
        """初始化简化状态空间模型"""
        # 这些参数需要根据UVa/Padova模型调整
        self.A = np.array([[0.8, -0.05, 0.02],  # 葡萄糖动态
                           [0.0, 0.9, 0.0],  # 胰岛素动态
                           [0.0, 0.0, 0.7]])  # 碳水化合物动态

        self.B = np.array([[0.0, -0.1],  # basal和bolus对葡萄糖的影响
                           [0.1, 0.3],  # basal和bolus对胰岛素的影响
                           [0.0, 0.0]])  # 无直接影响

        self.C = np.array([[1, 0, 0]])  # 只观测葡萄糖

    def _state_estimator(self, CGM, meal, last_action):
        """
        状态估计器 - 扩展卡尔曼滤波简化版
        Args:
            CGM: 当前血糖观测值
            meal: 当前碳水化合物摄入量
            last_action: 上一时刻的胰岛素输注
        """
        if self.last_CGM is None:
            self.x_est = np.array([CGM, 0.05, 0])
        else:
            # 预测步骤
            u = np.array([last_action.basal, last_action.bolus])
            self.x_est = self.A @ self.x_est + self.B @ u
            self.x_est[2] += meal  # 添加新摄入的碳水化合物

            # 更新步骤
            error = CGM - self.x_est[0]
            self.x_est += np.array([0.5 * error, 0.1 * error, 0])  # 简化的卡尔曼增益

        self.last_CGM = CGM
        return self.x_est

    def _mpc_optimization(self, x0, meal_announcement):
        """
        构建并求解MPC优化问题
        Args:
            x0: 初始状态
            meal_announcement: 未来预测的碳水化合物摄入
        Returns:
            optimal_basal: 基础率调整(U/h)
            optimal_bolus: 推注剂量(U)
        """
        nx = self.A.shape[0]
        nu = 2  # basal和bolus

        # 优化变量
        u_basal = Variable((1, self.M))
        u_bolus = Variable((1, self.M))
        x = Variable((nx, self.N + 1))

        # 初始状态约束
        constraints = [x[:, 0] == x0]

        # 系统动态约束
        for k in range(self.N):
            # 控制输入 (超过控制时域后保持最后值)
            basal = u_basal[:, min(k, self.M - 1)]
            bolus = u_bolus[:, min(k, self.M - 1)]
            u = np.vstack([basal, bolus])

            # 状态方程
            x_next = self.A @ x[:, k] + self.B @ u

            # 添加碳水化合物影响
            if k < len(meal_announcement):
                x_next[2] += meal_announcement[k]

            constraints += [x[:, k + 1] == x_next]

            # 输入约束
            constraints += [basal >= 0, basal <= self.basal_max,
                            bolus >= 0, bolus <= self.bolus_max]

            # 安全约束 (避免低血糖)
            constraints += [x[0, k + 1] >= self.safe_min]

        # 目标函数
        cost = 0
        for k in range(1, self.N + 1):
            cost += self.Q * square(x[0, k] - self.target)
            if k < self.M:
                cost += self.R_basal * square(u_basal[:, k])
                cost += self.R_bolus * square(u_bolus[:, k])

        # 构建并求解问题
        problem = Problem(Minimize(cost), constraints)
        problem.solve(solver=ECOS)

        if problem.status != OPTIMAL:
            print("MPC求解失败，使用安全基础率")
            return 10, 0

        return u_basal[0, 0].value, u_bolus[0, 0].value

    def policy(self, observation, reward, done, **info):
        '''
        MPC控制策略
        ----
        Inputs:
        observation - 包含CGM血糖值的命名元组
        reward      - 当前奖励值
        done        - 是否结束标志
        info        - 包含patient_name和sample_time等信息
        ----
        Output:
        action - 包含basal和bolus的命名元组
        '''
        CGM = observation.CGM
        meal = info.get('meal', 0)  # 当前碳水化合物摄入

        # 获取未来餐食信息 (简化处理，实际应用中需要更精确的预测)
        meal_announcement = [meal] + [0] * (self.N - 1)  # 假设只有当前时刻有餐食

        # 状态估计
        x0 = self._state_estimator(CGM, meal, self.last_action if hasattr(self, 'last_action') else None)

        # MPC优化
        basal, bolus = self._mpc_optimization(x0, meal_announcement)

        # 创建动作
        action = Action(basal=max(0, basal), bolus=max(0, bolus))
        self.last_action = action  # 保存当前动作用于下次状态估计

        return action

    def reset(self):
        '''重置控制器状态'''
        self.x_est = np.zeros(3)
        self.last_CGM = None
        if hasattr(self, 'last_action'):
            del self.last_action