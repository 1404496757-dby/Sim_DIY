from .base import Controller
from .base import Action
import numpy as np
import pandas as pd
import pkg_resources
import logging
import scipy.optimize as optimize

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

    def _objective_function(self, insulin_plan, initial_glucose, meal_prediction, steps, target):
        """
        MPC优化的目标函数

        参数:
        - insulin_plan: 胰岛素输注计划
        - initial_glucose: 初始血糖水平
        - meal_prediction: 预测的餐食摄入
        - steps: 预测步数
        - target: 目标血糖水平

        返回:
        - 目标函数值 (越小越好)
        """
        # 预测血糖水平
        glucose_pred = self._predict_glucose(initial_glucose, insulin_plan, meal_prediction, steps)

        # 计算与目标的偏差
        glucose_error = glucose_pred - target

        # 计算目标函数值
        # 1. 血糖偏离目标值的惩罚
        obj_value = np.sum(glucose_error ** 2)

        # 2. 低血糖惩罚 (更严重地惩罚低血糖)
        hypoglycemia_penalty = 10.0 * np.sum(np.maximum(0, 80 - glucose_pred))
        obj_value += hypoglycemia_penalty

        # 3. 胰岛素变化率惩罚 (平滑控制)
        if len(insulin_plan) > 1:
            insulin_rate_change = np.diff(insulin_plan)
            obj_value += 0.1 * np.sum(insulin_rate_change ** 2)

        return obj_value

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

        try:
            # 初始胰岛素计划 (使用基础率作为初始猜测)
            initial_insulin_plan = np.ones(ctrl_steps) * basal

            # 设置优化边界
            bounds = [(0, basal * 10) for _ in range(ctrl_steps)]

            # 使用scipy.optimize进行优化
            result = optimize.minimize(
                self._objective_function,
                initial_insulin_plan,
                args=(glucose, meal_prediction[:ctrl_steps], ctrl_steps, self.target),
                bounds=bounds,
                method='L-BFGS-B'
            )

            if result.success:
                # 获取最优胰岛素输注计划
                optimal_insulin_plan = result.x

                # 使用第一个控制动作
                optimal_insulin = optimal_insulin_plan[0]

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