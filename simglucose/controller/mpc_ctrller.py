import numpy as np
import cvxpy as cp
from .base import Controller, Action


class MyController(Controller):
    def __init__(self, insulin=1.0):
        self.insulin = insulin  # 初始胰岛素值
        self.pred_horizon = 10  # 预测步长
        self.dt = 5  # 采样时间（分钟）
        self.target_glucose = 110  # 目标血糖 (mg/dL)

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
            u <= 2.0,  # 胰岛素最大值约束
            x >= 70,  # 防止低血糖
            x <= 180  # 防止高血糖
        ]

        # 求解优化问题
        prob = cp.Problem(objective, constraints)
        prob.solve()

        # 返回计算出的第一步最优胰岛素值
        return max(0, min(2.0, u.value[0]))  # 约束在 0-2.0 U/min

    def reset(self):
        """重置控制器"""
        self.insulin = 1.0
