# my_nmpc_controller.py
from simglucose.controller.base import Controller, Action
import casadi as ca
import numpy as np
from simglucose.simulation.env import T1DSimEnv


class NMPCController(Controller):
    def __init__(self, prediction_horizon=6, control_horizon=3):
        # UVa/Padova模型参数（示例值，需根据实际模型调整）
        self.Ts = 3  # 采样时间（分钟）
        self.N = prediction_horizon  # 预测时域
        self.M = control_horizon  # 控制时域

        # 定义模型方程（需替换为UVa/Padova实际模型）
        self.setup_model()

        # 构建NMPC优化器
        self.setup_optimizer()

        # 状态初始化
        self.x0 = np.zeros(4)  # 假设模型为4状态（需确认实际状态维度）
        self.u_prev = 0

    def setup_model(self):
        """定义UVa/Padova离散非线性模型"""
        # 状态变量（示例，需根据实际模型调整）
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

        # 离散化（前向欧拉）
        self.f = ca.Function('f', [x, u], [x + self.Ts * dxdt])
        self.n_x = 4
        self.n_u = 1

    def setup_optimizer(self):
        """构建NMPC优化问题"""
        opti = ca.Opti()

        # 决策变量
        X = opti.variable(self.n_x, self.N + 1)  # 状态轨迹
        U = opti.variable(self.n_u, self.M)  # 控制序列

        # 参数
        x0 = opti.parameter(self.n_x)  # 初始状态
        G_ref = opti.parameter()  # 血糖参考值（110 mg/dL）

        # 目标函数
        J = 0
        for k in range(self.N):
            # 血糖跟踪误差项
            J += 100 * (X[0, k] - G_ref) ** 2
            # 控制量变化惩罚
            if k < self.M:
                J += 0.1 * U[0, k] ** 2

        # 动态约束
        for k in range(self.N):
            x_next = self.f(X[:, k], U[:, k] if k < self.M else U[:, -1])
            opti.subject_to(X[:, k + 1] == x_next)

        # 其他约束
        opti.subject_to(opti.bounded(70, X[0, :], 180))  # 血糖安全范围
        opti.subject_to(opti.bounded(0, U, 5))  # 胰岛素输注限制

        # 求解器设置
        opts = {'ipopt.print_level': 0, 'print_time': 0}
        opti.solver('ipopt', opts)

        self.opti = opti
        self.J = J
        self.sol = None

    def policy(self, observation, reward, done, **info):
        """NMPC控制策略"""
        # 获取当前血糖值（CGM测量）
        current_g = observation.CGM

        # 状态估计（简化处理，实际需要观测器）
        self.x0[0] = current_g  # 假设第一个状态是血糖

        # 设置优化问题参数
        self.opti.set_value(self.opti.p(G_ref), 110)
        self.opti.set_value(self.opti.p(x0), self.x0)

        # 求解优化问题
        try:
            sol = self.opti.solve()
            u_opt = sol.value(U)[0]
            self.u_prev = u_opt
        except:
            u_opt = self.u_prev  # 失败时使用上次解

        # 转换为basal/bolus（示例转换）
        return Action(basal=u_opt, bolus=0)

    def reset(self):
        self.u_prev = 0
        self.x0 = np.zeros(self.n_x)

# 使用示例
# patient = T1DSimEnv.get_patient_parameters('adult#001')
# controller = NMPCController(patient)