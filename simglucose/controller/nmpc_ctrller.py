# my_nmpc_controller.py
from simglucose.controller.base import Controller, Action
import casadi as ca
import numpy as np
from simglucose.simulation.env import T1DSimEnv


class NMPCController(Controller):
    def __init__(self, patient_model, prediction_horizon=6, control_horizon=3):
        self.Ts = 3  # 采样时间（分钟）
        self.N = prediction_horizon  # 预测时域
        self.M = control_horizon  # 控制时域

        self.setup_model()
        self.setup_optimizer()

        self.x0 = np.zeros(4)  # 根据实际模型调整状态维度
        self.u_prev = 0

    def setup_model(self):
        """定义UVa/Padova离散非线性模型"""
        # 状态变量（示例）
        x1 = ca.MX.sym('G_sub')  # 皮下葡萄糖
        x2 = ca.MX.sym('I_pl')  # 血浆胰岛素
        x3 = ca.MX.sym('G_liv')  # 肝脏葡萄糖
        x4 = ca.MX.sym('I_sc')  # 皮下胰岛素池

        x = ca.vertcat(x1, x2, x3, x4)
        u = ca.MX.sym('u')  # 胰岛素输注率

        # 示例模型方程（需替换为实际微分方程）
        dxdt = ca.vertcat(
            -0.05 * x1 + 0.01 * x3 - 0.001 * x2 + 0.1 * u,
            0.02 * x4 - 0.04 * x2,
            -0.01 * x3 + 0.005 * x1,
            0.1 * u - 0.05 * x4
        )

        self.f = ca.Function('f', [x, u], [x + self.Ts * dxdt])
        self.n_x = 4
        self.n_u = 1

    def setup_optimizer(self):
        """构建NMPC优化问题"""
        opti = ca.Opti()

        # 决策变量（保存为实例变量）
        self.X = opti.variable(self.n_x, self.N + 1)
        self.U = opti.variable(self.n_u, self.M)

        # 参数（保存为实例变量）
        self.x0_param = opti.parameter(self.n_x)
        self.G_ref = opti.parameter()

        # 目标函数
        J = 0
        for k in range(self.N):
            J += 100 * (self.X[0, k] - self.G_ref) ** 2
            if k < self.M:
                J += 0.1 * self.U[0, k] ** 2

        # 动态约束
        for k in range(self.N):
            u = self.U[:, k] if k < self.M else self.U[:, -1]
            x_next = self.f(self.X[:, k], u)
            opti.subject_to(self.X[:, k + 1] == x_next)

        # 其他约束
        opti.subject_to(opti.bounded(70, self.X[0, :], 180))
        opti.subject_to(opti.bounded(0, self.U, 5))

        # 初始状态约束
        opti.subject_to(self.X[:, 0] == self.x0_param)

        # 求解器设置
        opts = {'ipopt.print_level': 0, 'print_time': 0}
        opti.solver('ipopt', opts)

        self.opti = opti
        self.cost = J
        self.sol = None

    def policy(self, observation, reward, done, **info):
        """NMPC控制策略"""
        current_g = observation.CGM

        # 状态估计（简化处理）
        self.x0[0] = current_g

        # 设置参数值（通过实例变量访问）
        self.opti.set_value(self.G_ref, 110)
        self.opti.set_value(self.x0_param, self.x0)

        try:
            sol = self.opti.solve()
            u_opt = sol.value(self.U)[0][0]  # 访问第一个控制量
            self.u_prev = u_opt
        except:
            u_opt = self.u_prev

        return Action(basal=u_opt, bolus=0)

    def reset(self):
        self.u_prev = 0
        self.x0 = np.zeros(self.n_x)