from .base import Controller
from .base import Action
import numpy as np
import pandas as pd
import pkg_resources
import logging
import cvxpy as cp
from scipy.interpolate import interp1d

logger = logging.getLogger(__name__)
CONTROL_QUEST = pkg_resources.resource_filename('simglucose',
                                                'params/Quest.csv')
PATIENT_PARA_FILE = pkg_resources.resource_filename(
    'simglucose', 'params/vpatient_params.csv')


class MPCController(Controller):
    """
    模型预测控制器(MPC)用于1型糖尿病患者的血糖控制。
    该控制器使用简化的血糖动力学模型预测未来血糖水平，
    并通过求解优化问题来确定最佳的胰岛素输注量。
    """

    def __init__(self, target=140, prediction_horizon=6, control_horizon=3):
        """
        初始化MPC控制器

        参数:
        - target: 目标血糖水平 (mg/dL)
        - prediction_horizon: 预测时域长度 (小时)
        - control_horizon: 控制时域长度 (小时)
        """
        self.quest = pd.read_csv(CONTROL_QUEST)
        self.patient_params = pd.read_csv(PATIENT_PARA_FILE)
        self.target = target

        # MPC参数
        self.prediction_horizon = prediction_horizon  # 预测时域 (小时)
        self.control_horizon = control_horizon  # 控制时域 (小时)

        # 状态变量
        self.prev_glucose = None
        self.insulin_history = []
        self.glucose_history = []
        self.meal_history = []
        self.time_step = 0

        # 模型参数 (将根据患者进行调整)
        self.insulin_sensitivity = 0.05  # 胰岛素敏感性
        self.carb_sensitivity = 0.2  # 碳水化合物敏感性
        self.glucose_decay = 0.98  # 血糖自然衰减率

    def policy(self, observation, reward, done, **kwargs):
        """实现MPC控制策略"""
        sample_time = kwargs.get('sample_time', 1)  # 采样时间 (分钟)
        pname = kwargs.get('patient_name')
        meal = kwargs.get('meal', 0)  # 餐食 (g/min)

        # 获取当前血糖值
        current_glucose = observation.CGM

        # 更新历史数据
        self.glucose_history.append(current_glucose)
        self.meal_history.append(meal)

        # 保持历史数据长度适中
        max_history = 24 * 60  # 24小时的数据
        if len(self.glucose_history) > max_history:
            self.glucose_history = self.glucose_history[-max_history:]
            self.meal_history = self.meal_history[-max_history:]
            self.insulin_history = self.insulin_history[-max_history:]

        # 调整模型参数 (基于患者特征)
        self._adjust_model_params(pname)

        # 计算MPC控制动作
        action = self._mpc_policy(pname, meal, current_glucose, sample_time)

        # 更新胰岛素历史
        self.insulin_history.append(action.basal + action.bolus)
        self.time_step += 1

        return action

    def _adjust_model_params(self, name):
        """根据患者特征调整模型参数"""
        if any(self.quest.Name.str.match(name)):
            quest = self.quest[self.quest.Name.str.match(name)]
            params = self.patient_params[self.patient_params.Name.str.match(name)]

            # 根据患者参数调整模型
            CR = quest.CR.values.item()  # 碳水化合物比例
            CF = quest.CF.values.item()  # 校正因子

            # 调整模型参数
            self.insulin_sensitivity = 1 / (CF * 50)  # 根据校正因子调整胰岛素敏感性
            self.carb_sensitivity = 1 / (CR * 10)  # 根据碳水化合物比例调整碳水敏感性
        else:
            # 使用默认参数
            self.insulin_sensitivity = 0.05
            self.carb_sensitivity = 0.2
            self.glucose_decay = 0.98

    def _predict_glucose(self, initial_glucose, insulin_plan, meal_prediction, steps):
        """
        使用简化模型预测未来血糖水平

        参数:
        - initial_glucose: 初始血糖水平
        - insulin_plan: 计划的胰岛素输注量
        - meal_prediction: 预测的餐食摄入
        - steps: 预测步数

        返回:
        - 预测的血糖序列
        """
        glucose_pred = np.zeros(steps)
        glucose_pred[0] = initial_glucose

        for i in range(1, steps):
            # 简化的血糖动力学模型
            glucose_decay_effect = glucose_pred[i - 1] * self.glucose_decay
            insulin_effect = insulin_plan[i - 1] * self.insulin_sensitivity * 100
            meal_effect = meal_prediction[i - 1] * self.carb_sensitivity * 10

            # 计算下一步血糖
            glucose_pred[i] = glucose_decay_effect - insulin_effect + meal_effect

            # 确保血糖不会低于一个安全值
            glucose_pred[i] = max(glucose_pred[i], 70)

        return glucose_pred

    def _mpc_policy(self, name, meal, glucose, sample_time):
        """
        实现MPC控制策略

        参数:
        - name: 患者名称
        - meal: 当前餐食摄入 (g/min)
        - glucose: 当前血糖水平 (mg/dL)
        - sample_time: 采样时间 (分钟)

        返回:
        - Action对象，包含基础率和大剂量
        """
        # 获取患者特定参数
        if any(self.quest.Name.str.match(name)):
            quest = self.quest[self.quest.Name.str.match(name)]
            params = self.patient_params[self.patient_params.Name.str.match(name)]
            u2ss = params.u2ss.values.item()  # unit: pmol/(L*kg)
            BW = params.BW.values.item()  # unit: kg
        else:
            quest = pd.DataFrame([['Average', 1 / 15, 1 / 50, 50, 30]],
                                 columns=['Name', 'CR', 'CF', 'TDI', 'Age'])
            u2ss = 1.43  # unit: pmol/(L*kg)
            BW = 57.0  # unit: kg

        # 计算基础胰岛素率
        basal = u2ss * BW / 6000  # unit: U/min

        # 设置预测和控制步长
        pred_steps = int(self.prediction_horizon * 60 / sample_time)  # 预测步数
        ctrl_steps = int(self.control_horizon * 60 / sample_time)  # 控制步数

        # 简单的餐食预测 (假设未来没有餐食，除非当前有餐食)
        meal_prediction = np.zeros(pred_steps)
        if meal > 0:
            # 如果当前有餐食，假设会持续一段时间
            meal_duration = int(30 / sample_time)  # 假设餐食持续30分钟
            meal_prediction[:min(meal_duration, pred_steps)] = meal

        # 使用CVXPY求解MPC优化问题
        try:
            # 定义优化变量 (胰岛素输注计划)
            insulin_var = cp.Variable(ctrl_steps)

            # 设置约束条件
            constraints = [
                insulin_var >= 0,  # 胰岛素不能为负
                insulin_var <= basal * 10  # 胰岛素上限
            ]

            # 构建完整的胰岛素计划 (控制时域后保持最后一个值)
            # insulin_plan = cp.hstack([insulin_var, cp.repeat(insulin_var[-1], pred_steps - ctrl_steps)])
            insulin_plan_extension = []
            for _ in range(pred_steps - ctrl_steps):
                insulin_plan_extension.append(insulin_var[-1])

            if len(insulin_plan_extension) > 0:
                insulin_plan = cp.hstack([insulin_var, *insulin_plan_extension])
            else:
                insulin_plan = insulin_var

            # 预测血糖水平
            initial_glucose = glucose

            # 定义目标函数
            # 1. 血糖偏离目标值的惩罚
            glucose_pred = self._predict_glucose(initial_glucose, insulin_plan.value, meal_prediction, pred_steps)
            glucose_error = glucose_pred - self.target

            # 使用二次型目标函数
            objective = cp.sum_squares(glucose_error)

            # 2. 添加胰岛素变化率惩罚 (平滑控制)
            if len(self.insulin_history) > 0:
                last_insulin = self.insulin_history[-1]
                insulin_rate_change = cp.diff(cp.hstack([last_insulin, insulin_var]))
                objective += 0.1 * cp.sum_squares(insulin_rate_change)

            # 3. 低血糖惩罚 (更严重地惩罚低血糖)
            hypoglycemia_penalty = 10.0 * cp.sum(cp.maximum(0, 80 - glucose_pred))
            objective += hypoglycemia_penalty

            # 定义并求解问题
            prob = cp.Problem(cp.Minimize(objective), constraints)
            prob.solve(solver=cp.OSQP, verbose=False)

            # 检查求解状态
            if prob.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                # 获取最优胰岛素输注量 (仅使用第一个控制动作)
                optimal_insulin = insulin_var.value[0]

                # 将总胰岛素分为基础率和大剂量
                if meal > 0:
                    # 如果有餐食，计算大剂量
                    bolus_component = (meal * sample_time) / quest.CR.values.item()
                    correction_component = max(0, (glucose - self.target) / quest.CF.values.item())

                    # 总大剂量 (单位: U)
                    total_bolus = min(bolus_component + correction_component, optimal_insulin * sample_time)

                    # 转换为速率 (U/min)
                    bolus = total_bolus / sample_time

                    # 剩余部分作为基础率
                    adjusted_basal = max(0, optimal_insulin - bolus)
                else:
                    # 无餐食时，全部作为基础率
                    adjusted_basal = optimal_insulin
                    bolus = 0
            else:
                # 优化失败，使用传统方法
                logger.warning("MPC优化失败，使用传统基础-玻尔斯方法")
                adjusted_basal = basal

                if meal > 0:
                    bolus_amount = ((meal * sample_time) / quest.CR.values +
                                    (glucose > 150) * (glucose - self.target) / quest.CF.values).item()
                    bolus = bolus_amount / sample_time
                else:
                    bolus = 0

        except Exception as e:
            # 出现异常时，使用传统方法
            logger.error(f"MPC计算出错: {str(e)}")
            adjusted_basal = basal

            if meal > 0:
                bolus_amount = ((meal * sample_time) / quest.CR.values +
                                (glucose > 150) * (glucose - self.target) / quest.CF.values).item()
                bolus = bolus_amount / sample_time
            else:
                bolus = 0

        return Action(basal=adjusted_basal, bolus=bolus)

    def reset(self):
        """重置控制器状态"""
        self.prev_glucose = None
        self.insulin_history = []
        self.glucose_history = []
        self.meal_history = []
        self.time_step = 0