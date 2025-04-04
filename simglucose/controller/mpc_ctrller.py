from cvxpy import *
import scipy.linalg as la


class MPCController:
    def __init__(self, patient, prediction_horizon=10, control_horizon=5):
        self.patient = patient
        self.N = prediction_horizon
        self.M = control_horizon

        # 获取线性化模型（这里需要根据UVa/Padova模型具体实现）
        self.A, self.B, self.C = self._linearize_model()

        # 目标血糖值 (mg/dL)
        self.G_target = 120

        # 控制权重矩阵
        self.Q = np.diag([1.0])  # 状态误差权重
        self.R = np.diag([0.1])  # 控制变化权重

        # 约束条件
        self.u_min = 0  # 最小胰岛素输注率
        self.u_max = 5  # 最大胰岛素输注率

    def _linearize_model(self):
        """线性化UVa/Padova模型"""
        # 这里需要根据模型具体实现线性化
        # 简化示例 - 实际需要根据模型方程计算雅可比矩阵
        A = np.eye(12) * 0.9  # 示例状态矩阵
        B = np.zeros((12, 1))
        B[8, 0] = 1.0  # 胰岛素输入主要影响血浆胰岛素状态
        C = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])  # 只观测血糖

        return A, B, C

    def _build_optimization_problem(self, x0):
        """构建MPC优化问题"""
        nx = self.A.shape[0]  # 状态维度
        nu = self.B.shape[1]  # 输入维度
        ny = self.C.shape[0]  # 输出维度

        # 优化变量
        u = Variable((nu, self.M))
        x = Variable((nx, self.N + 1))
        y = Variable((ny, self.N))

        # 初始状态约束
        constraints = [x[:, 0] == x0]

        # 系统动态约束
        for k in range(self.N):
            # 状态方程
            constraints += [x[:, k + 1] == self.A @ x[:, k] + self.B @ u[:, min(k, self.M - 1)]]

            # 输出方程
            constraints += [y[:, k] == self.C @ x[:, k]]

            # 输入约束
            constraints += [u[:, min(k, self.M - 1)] >= self.u_min,
                            u[:, min(k, self.M - 1)] <= self.u_max]

        # 目标函数
        cost = 0
        for k in range(self.N):
            cost += quad_form(y[:, k] - self.G_target, self.Q)
            if k < self.M:
                cost += quad_form(u[:, k], self.R)

        # 构建问题
        problem = Problem(Minimize(cost), constraints)

        return problem, u

    def __call__(self, observation, past_actions=None):
        """执行MPC控制"""
        # 从观测中提取状态（简化处理，实际需要状态估计）
        CGM = observation.CGM
        x0 = np.zeros((12,))  # 实际应用中需要完整状态估计
        x0[0] = CGM  # 假设第一个状态是血糖

        # 构建并求解MPC问题
        problem, u_var = self._build_optimization_problem(x0)
        problem.solve(solver=ECOS)

        if problem.status != OPTIMAL:
            print("MPC求解失败，使用基础率")
            return self.patient._basal

        # 返回第一个控制动作
        return u_var[0, 0].value