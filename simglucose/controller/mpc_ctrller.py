from .base import Controller
from .base import Action
import numpy as np
import pandas as pd
import pkg_resources
import logging
from scipy.optimize import minimize
from scipy.linalg import solve_discrete_are

logger = logging.getLogger(__name__)
CONTROL_QUEST = pkg_resources.resource_filename('simglucose', 'params/Quest.csv')
PATIENT_PARA_FILE = pkg_resources.resource_filename('simglucose', 'params/vpatient_params.csv')


class MPCController(Controller):
    """
    MPC Controller for artificial pancreas using UVA/Padova simulator
    """

    def __init__(self, target=140, prediction_horizon=12, control_horizon=6):
        self.quest = pd.read_csv(CONTROL_QUEST)
        self.patient_params = pd.read_csv(PATIENT_PARA_FILE)
        self.target = target
        self.N = prediction_horizon  # 预测时域
        self.Nu = control_horizon  # 控制时域
        self.dt = 5  # 分钟(与模拟器步长匹配)

        # 初始化模型参数(将在第一次调用policy时设置)
        self.A = None
        self.B = None
        self.Q = None
        self.R = None
        self.P = None
        self.u_max = None
        self.u_min = 0.0
        self.initialized = False

    def _initialize_model(self, name):
        """根据患者名称初始化模型参数"""
        if any(self.quest.Name.str.match(name)):
            quest = self.quest[self.quest.Name.str.match(name)]
            params = self.patient_params[self.patient_params.Name.str.match(name)]
            u2ss = params.u2ss.values.item()  # unit: pmol/(L*kg)
            BW = params.BW.values.item()  # unit: kg
            # 从参数中提取更多需要的参数...
        else:
            quest = pd.DataFrame([['Average', 1 / 15, 1 / 50, 50, 30]],
                                 columns=['Name', 'CR', 'CF', 'TDI', 'Age'])
            u2ss = 1.43  # unit: pmol/(L*kg)
            BW = 57.0  # unit: kg

        # 设置安全约束
        self.u_max = 2.0  # U/hr (最大基础率)

        # 创建简化的Bergman最小模型状态空间表示
        p1 = 0.028735  # 患者参数，可从模拟器参数中获取更精确的值
        p2 = 0.028344
        p3 = 5.035e-5

        self.A = np.array([[1 - p1 * self.dt, -p2 * self.dt],
                           [0, 1 - p3 * self.dt]])
        self.B = np.array([[0], [self.dt / (BW * 100)]])  # 转换为dL

        # 确保矩阵维度正确
        self.A = np.reshape(self.A, (2, 2))
        self.B = np.reshape(self.B, (2, 1))

        # MPC权重矩阵
        self.Q = np.diag([1.0, 0.01])  # 血糖权重高，胰岛素变化权重低
        self.R = np.array([[0.1]])  # 控制动作惩罚

        # 终端代价(通过Riccati方程计算)
        self.P = solve_discrete_are(self.A, self.B, self.Q, self.R)

        self.initialized = True
        return BW, quest

    def predict_glucose(self, x0, insulin_sequence):
        """预测血糖轨迹"""
        x_pred = np.zeros((self.N + 1, 2))
        x_pred[0] = x0

        for k in range(self.N):
            if k < len(insulin_sequence):
                u = insulin_sequence[k]
            else:
                u = 0.0  # 超出控制时域后假设零输入

            # 确保矩阵乘法维度正确
            x_pred[k + 1] = (self.A @ x_pred[k].reshape(-1, 1) + self.B * u).flatten()

        return x_pred

    def cost_function(self, u_flat, x0):
        """MPC代价函数"""
        u_seq = u_flat.reshape(-1, 1)
        x_pred = self.predict_glucose(x0, u_seq)

        cost = 0
        for k in range(self.N):
            # 血糖误差(相对于目标)
            glucose_error = x_pred[k][0] - self.target
            cost += glucose_error ** 2 * self.Q[0, 0] + u_seq[k] ** 2 * self.R[0, 0]

        # 终端代价
        terminal_error = x_pred[self.N][0] - self.target
        cost += terminal_error ** 2 * self.P[0, 0]

        return cost

    def policy(self, observation, reward, done, **kwargs):
        sample_time = kwargs.get('sample_time', 1)
        pname = kwargs.get('patient_name')
        meal = kwargs.get('meal')  # unit: g/min

        # 第一次调用时初始化模型
        if not self.initialized:
            BW, quest = self._initialize_model(pname)

        # 当前状态 (血糖和胰岛素作用)
        current_glucose = observation.CGM
        x0 = np.array([current_glucose, 0.0])  # 假设胰岛素作用初始为0

        # 优化问题设置
        def constraint_func(u_flat):
            u_seq = u_flat.reshape(-1, 1)
            x_pred = self.predict_glucose(x0, u_seq)
            # 可以添加状态约束(如血糖不能低于某个值)
            return x_pred[:, 0]  # 返回所有预测的血糖值

        # 初始猜测(患者基础率)
        if any(self.quest.Name.str.match(pname)):
            params = self.patient_params[self.patient_params.Name.str.match(pname)]
            u2ss = params.u2ss.values.item()
            BW = params.BW.values.item()
            u0 = np.ones(self.Nu) * u2ss * BW / 6000  # U/min
        else:
            u0 = np.ones(self.Nu) * 0.01  # 默认基础率

        # 约束条件
        bounds = [(self.u_min, self.u_max)] * self.Nu
        constraints = {
            'type': 'ineq',
            'fun': lambda u: constraint_func(u) - 70  # 血糖>70 mg/dL
        }

        # 如果有餐食，添加推注
        if meal > 0:
            logger.info('Calculating bolus for meal...')
            if any(self.quest.Name.str.match(pname)):
                quest = self.quest[self.quest.Name.str.match(pname)]
                bolus = ((meal * sample_time) / quest.CR.values +
                         (current_glucose - self.target) / quest.CF.values).item()
            else:
                bolus = (meal * sample_time) / 15  # 默认碳水化合物比例

            # 将推注添加到初始控制序列
            u0[0] += bolus / sample_time  # 转换为U/min

            # 限制推注不超过最大值
            u0 = np.clip(u0, self.u_min, self.u_max)

        # 优化
        res = minimize(self.cost_function, u0, args=(x0,),
                       bounds=bounds, constraints=constraints,
                       method='SLSQP')

        if not res.success:
            logger.warning(f"MPC optimization failed: {res.message}")
            # 失败时返回基础率
            basal = u0[0] if meal == 0 else u0[0] - bolus / sample_time
            return Action(basal=basal, bolus=0)

        # 提取最优控制序列
        u_opt = res.x.reshape((self.Nu, 1))

        # 仅应用第一个控制输入 (根据MPC原理)
        basal = u_opt[0, 0]

        # 如果有餐食，将推注部分分离出来
        bolus = 0
        if meal > 0:
            if any(self.quest.Name.str.match(pname)):
                params = self.patient_params[self.patient_params.Name.str.match(pname)]
                u2ss = params.u2ss.values.item()
                BW = params.BW.values.item()
                basal_normal = u2ss * BW / 6000
            else:
                basal_normal = 0.01

            bolus = max(0, basal - basal_normal) * sample_time
            basal = basal_normal

        return Action(basal=basal, bolus=bolus)

    def reset(self):
        """重置控制器状态"""
        self.initialized = False
