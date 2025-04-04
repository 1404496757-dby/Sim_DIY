import cvxpy as cp

class MPCController(Controller):
    def __init__(self, init_state, prediction_horizon=5, control_horizon=2):
        super().__init__(init_state)
        self.prediction_horizon = prediction_horizon
        self.control_horizon = control_horizon
        # 定义模型参数
        self.A = np.array([[1]])  # 状态转移矩阵
        self.B = np.array([[1]])  # 控制输入矩阵
        self.Q = np.array([[1]])  # 状态权重矩阵
        self.R = np.array([[1]])  # 控制输入权重矩阵
        self.reference = 140  # 参考血糖值

    def policy(self, observation, reward, done, **info):
        current_cgm = observation.CGM
        # 定义优化变量
        x = cp.Variable((1, self.prediction_horizon + 1))
        u = cp.Variable((1, self.control_horizon))

        # 定义目标函数
        cost = 0
        for t in range(self.prediction_horizon):
            cost += cp.quad_form(x[:, t + 1] - self.reference, self.Q)
            if t < self.control_horizon:
                cost += cp.quad_form(u[:, t], self.R)

        # 定义约束条件
        constraints = []
        constraints.append(x[:, 0] == current_cgm)
        for t in range(self.prediction_horizon):
            if t < self.control_horizon:
                constraints.append(x[:, t + 1] == self.A @ x[:, t] + self.B @ u[:, t])
            else:
                constraints.append(x[:, t + 1] == self.A @ x[:, t])

        # 定义优化问题
        prob = cp.Problem(cp.Minimize(cost), constraints)

        # 求解优化问题
        prob.solve()

        # 获取最优控制输入
        if prob.status == cp.OPTIMAL:
            basal = u.value[0, 0]
            bolus = 0
        else:
            basal = 0
            bolus = 0

        action = Action(basal=basal, bolus=bolus)
        return action

    def reset(self):
        self.state = self.init_state