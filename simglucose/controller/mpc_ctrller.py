import numpy as np
import cvxpy as cp
from simglucose.simulation.env import T1DSimEnv
from simglucose.controller.base import Controller, Action
from simglucose.simulation.sim_engine import SimObj, sim
from simglucose.simulation.scenario import CustomScenario


class MPCController(Controller):
    def __init__(self, patient_model=None, prediction_horizon=10, dt=3, target_glucose=120):
        """
        MPC控制器初始化

        参数:
            patient_model: 患者模型参数 (可选)
            prediction_horizon: 预测时域长度
            dt: 控制时间间隔 (分钟)
            target_glucose: 目标血糖值 (mg/dL)
        """
        self.pred_horizon = prediction_horizon
        self.dt = dt
        self.target = target_glucose
        self.insulin = 0  # 当前胰岛素输注速率

        # 患者特定参数 (默认使用成人#001参数)
        self.patient_params = patient_model or {
            'p1': 0.05,  # 葡萄糖自身清除率
            'p2': 0.08,  # 胰岛素敏感性衰减率
            'p3': 5e-6,  # 胰岛素对葡萄糖的影响
            'p4': 0.1  # 胰岛素清除率
        }

        # 安全约束
        self.insulin_min = 0  # 最小胰岛素速率 (U/h)
        self.insulin_max = 5  # 最大胰岛素速率 (ICU典型值)
        self.glucose_min = 70  # 低血糖阈值
        self.glucose_max = 250  # 高血糖阈值

    def policy(self, observation, reward, done, **info):
        """主控制策略"""
        current_glucose = observation.CGM
        optimal_insulin = self._mpc_optimization(current_glucose)

        # 应用安全限制
        self.insulin = np.clip(optimal_insulin, self.insulin_min, self.insulin_max)
        return Action(basal=self.insulin, bolus=0)

    def _mpc_optimization(self, current_glucose):
        """执行MPC优化计算"""
        N = self.pred_horizon

        # 定义优化变量
        u = cp.Variable(N)  # 胰岛素输入 (U/h)
        G = cp.Variable(N + 1)  # 血糖预测 (mg/dL)
        X = cp.Variable(N + 1)  # 胰岛素敏感性
        I = cp.Variable(N + 1)  # 血浆胰岛素浓度 (mU/L)

        # 初始条件
        constraints = [
            G[0] == current_glucose,
            X[0] == 0,  # 初始胰岛素敏感性
            I[0] == self.insulin * 1000 / 60  # 转换U/h→mU/min→mU/L
        ]

        # 系统动力学 (离散化Bergman最小模型)
        for k in range(N):
            constraints += [
                G[k + 1] == G[k] - self.patient_params['p1'] * G[k] * self.dt - self.patient_params['p3'] * X[k] * G[
                    k] * self.dt,
                X[k + 1] == X[k] - self.patient_params['p2'] * X[k] * self.dt + self.patient_params['p3'] * I[
                    k] * self.dt,
                I[k + 1] == I[k] - self.patient_params['p4'] * I[k] * self.dt + (u[k] * 1000 / 60) * self.dt
            ]

        # 目标函数: 最小化血糖偏差 + 控制输入变化
        objective = cp.Minimize(
            cp.sum_squares(G - self.target) + 0.01 * cp.sum_squares(u)
        )

        # 添加约束
        constraints += [
            u >= self.insulin_min,
            u <= self.insulin_max,
            G >= self.glucose_min,
            G <= self.glucose_max
        ]

        # 求解优化问题
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.ECOS)

        if prob.status != cp.OPTIMAL:
            print("MPC求解失败，使用上次输入")
            return self.insulin

        return u.value[0]  # 仅返回第一个控制输入

    def reset(self):
        """重置控制器状态"""
        self.insulin = 0


# ===================== 仿真测试 =====================
if __name__ == "__main__":
    # 创建仿真环境 (选择成人患者)
    patient = "adult#001"
    env = T1DSimEnv(patient)

    # 创建餐食场景 (7:00吃50g CHO)
    scenario = CustomScenario(start_time=0, meals=[(7 * 60, 50)])

    # 初始化MPC控制器
    controller = MPCController(
        prediction_horizon=10,
        dt=3,
        target_glucose=120
    )

    # 运行仿真
    sim_obj = SimObj(env, controller, scenario)
    results = sim(sim_obj, duration=24 * 60)  # 仿真24小时

    # 绘制结果
    from simglucose.analysis.report import report

    report(results, title="MPC控制仿真结果")