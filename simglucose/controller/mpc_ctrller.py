import numpy as np
import cvxpy as cp
from .base import Controller, Action


class MPCController(Controller):
    def __init__(self, prediction_horizon=10, dt=3, target_glucose=160):
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
        使用MPC计算最佳胰岛素剂量，基于Bergman最小模型
        """
        N = self.pred_horizon  # 预测步长
        u = cp.Variable(N)  # 胰岛素输入变量
        G = cp.Variable(N + 1)  # 血糖浓度
        X = cp.Variable(N + 1)  # 胰岛素敏感性
        I = cp.Variable(N + 1)  # 胰岛素浓度
        G0 = current_cgm  # 初始血糖值
        X0, I0 = 0, 0  # 初始状态假设

        # 约束条件
        constraints = [G[0] == G0, X[0] == X0, I[0] == I0]

        # 使用完整Bergman模型进行预测
        for i in range(N):
            constraints += [
                G[i + 1] == G[i] + self.dt * (-self.p1 * G[i] - X[i] * G[i]),
                X[i + 1] == X[i] + self.dt * (-self.p2 * X[i] + self.p3 * I[i]),
                I[i + 1] == I[i] + self.dt * (-self.p4 * I[i] + u[i])
            ]

        # 目标函数：最小化血糖偏差 + 胰岛素使用
        objective = cp.Minimize(
            cp.sum_squares(G - self.target_glucose) + 0.1 * cp.sum_squares(u)
        )

        # 约束条件
        constraints += [
            u >= 0,  # 胰岛素不能为负
            u <= 6,  # ICU 规定最大6U/h
            G >= 70,  # 防止低血糖
            G <= 250  # 防止高血糖
        ]

        # 根据ICU方案调整胰岛素
        if self.prev_glucose is not None:
            if current_cgm > 140 and current_cgm - self.prev_glucose >= 18:
                constraints.append(u[0] >= self.insulin + 1)
            elif current_cgm > 140 and self.prev_glucose - current_cgm >= 18:
                constraints.append(u[0] <= self.insulin - 1)

        # 求解优化问题
        prob = cp.Problem(objective, constraints)
        prob.solve()

        # 选择符合ICU方案的最优胰岛素速率
        optimal_insulin = max(0, min(6, u.value[0]))  # 限制在 0-6 U/h
        for limit in reversed(self.insulin_limits):
            if optimal_insulin >= limit:
                optimal_insulin = limit
                break

        self.prev_glucose = current_cgm  # 记录上次血糖
        return optimal_insulin  # 返回最优胰岛素值

    def reset(self):
        """重置控制器"""
        self.insulin = 0
        self.prev_glucose = None
