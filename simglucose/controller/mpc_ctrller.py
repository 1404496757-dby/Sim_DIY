import numpy as np
import cvxpy as cp
from .base import Controller, Action


class MPC_ICU_Controller(Controller):
    def __init__(self, prediction_horizon=10, dt=5,target_glucose = 160):
        self.insulin = 0  # 初始胰岛素值
        self.prev_glucose = None  # 记录上次血糖值
        self.pred_horizon = prediction_horizon  # 预测步长
        self.dt = dt  # 采样时间（分钟）
        self.target_glucose = target_glucose  # 目标血糖值（ICU 目标 7.8-10.0mmol/L, 取 160mg/dL）
        self.insulin_limits = [0, 2, 4, 5, 6]  # ICU 规定的胰岛素速率范围

    def policy(self, observation, reward, done, **info):
        """
        计算最优的胰岛素剂量（basal）基于 MPC 方法
        """
        current_cgm = observation.CGM  # 读取 CGM 传感器的血糖值
        optimal_insulin = self.mpc_controller(current_cgm)
        self.insulin = optimal_insulin
        return Action(basal=self.insulin, bolus=0)

    def mpc_controller(self, current_cgm):
        """
        使用 MPC 计算最佳胰岛素注射量
        """
        N = self.pred_horizon  # 预测步长
        u = cp.Variable(N)  # 胰岛素输入变量
        x = cp.Variable(N + 1)  # 预测血糖值
        x0 = current_cgm  # 初始血糖值

        # 动力学模型（简化版本）
        A = 1.0  # 血糖变化因子
        B = -0.5  # 胰岛素对血糖影响系数
        x_pred = [x0]
        for i in range(N):
            x_pred.append(A * x_pred[i] + B * u[i])

        # 目标函数（最小化血糖偏差 + 胰岛素使用量）
        objective = cp.Minimize(
            cp.sum_squares(x - self.target_glucose) + 0.1 * cp.sum_squares(u)
        )

        # 约束条件
        constraints = [
            x[0] == x0,
            x[1:] == A * x[:-1] + B * u,
            u >= 0,  # 胰岛素不能为负
            u <= 6,  # ICU 规定最大值 6U/h
            x >= 70,  # 防止低血糖
            x <= 250  # 防止过高血糖
        ]

        # ICU 胰岛素调整规则（与上次测量值对比）
        if self.prev_glucose is not None:
            if current_cgm > 140 and current_cgm - self.prev_glucose >= 18:
                constraints.append(u[0] >= self.insulin + 1)  # 增加 1U
            elif current_cgm > 140 and self.prev_glucose - current_cgm >= 18:
                constraints.append(u[0] <= self.insulin - 1)  # 减少 1U

        # 求解优化问题
        prob = cp.Problem(objective, constraints)
        prob.solve()

        # 选择符合 ICU 方案的最优胰岛素速率
        optimal_insulin = max(0, min(6, u.value[0]))  # 限制在 0-6 U/h
        for limit in reversed(self.insulin_limits):
            if optimal_insulin >= limit:
                optimal_insulin = limit
                break

        self.prev_glucose = current_cgm  # 更新血糖记录
        return optimal_insulin  # 返回最优胰岛素值

    def reset(self):
        """重置控制器"""
        self.insulin = 0
        self.prev_glucose = None
