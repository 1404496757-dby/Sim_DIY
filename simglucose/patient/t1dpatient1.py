from simglucose.patient.base import Patient
import numpy as np
from scipy.integrate import ode
import pandas as pd
from collections import namedtuple
import logging
import pkg_resources
import datetime

logger = logging.getLogger(__name__)

Action = namedtuple("patient_action", ["CHO", "insulin"])
Observation = namedtuple("observation", ["Gsub", "last_check_time", "next_check_time"])

PATIENT_PARA_FILE = pkg_resources.resource_filename(
    "simglucose", "params/vpatient_params.csv"
)

# ICU血糖控制常规参数
ICU_BG_TARGET_LOW = 7.8  # mmol/L (140 mg/dL)
ICU_BG_TARGET_HIGH = 10.0  # mmol/L (180 mg/dL)
ICU_BG_CRITICAL_LOW = 4.0  # mmol/L (72 mg/dL)
ICU_BG_SAFE = 6.1  # mmol/L (110 mg/dL)

# 胰岛素静脉泵参数
INSULIN_PUMP_CONCENTRATION = 1.0  # U/ml (40U in 40ml)

# 血糖监测时间点（小时）
MONITORING_TIMES_REGULAR = [6, 12, 18, 24]  # 每6小时监测一次
MONITORING_TIMES_MEAL = [-1, 5, 11, 17, 23]  # 进食前1小时及晚12点
MONITORING_TIMES_INTENSIVE = list(range(0, 24))  # 每小时监测一次


class ICUPatient(Patient):
    SAMPLE_TIME = 1  # min
    EAT_RATE = 5  # g/min CHO

    def __init__(self, params, init_state=None, random_init_bg=False, seed=None, t0=0,
                 patient_type="post_surgery", nutrition_type="fasting"):
        """
        ICUPatient constructor.
        Inputs:
            - params: a pandas sequence
            - init_state: customized initial state.
              If not specified, load the default initial state in
              params.iloc[2:15]
            - t0: simulation start time, it is 0 by default
            - patient_type: "post_surgery", "oral_diet", "nutrition_support", "special"
            - nutrition_type: "fasting", "oral_diet", "enteral", "parenteral"
        """
        self._params = params
        self._init_state = init_state
        self.random_init_bg = random_init_bg
        self._seed = seed
        self.t0 = t0

        # ICU相关参数
        self.patient_type = patient_type
        self.nutrition_type = nutrition_type
        self.icu_admission_time = datetime.datetime.now()
        self.last_check_time = None
        self.next_check_time = None
        self.monitoring_frequency = self._determine_monitoring_frequency()
        self.stable_bg_days = 0  # 连续稳定血糖的天数

        # 胰岛素治疗相关
        self.insulin_pump_active = False
        self.insulin_pump_rate = 0  # U/h
        self.last_bg_value = None
        self.last_insulin_adjustment_time = None

        self.reset()

    def _determine_monitoring_frequency(self):
        """确定血糖监测频率"""
        if self.patient_type == "special" or (
                datetime.datetime.now() - self.icu_admission_time).total_seconds() / 3600 < 24:
            return "intensive"  # 每1-2小时
        elif self.patient_type == "post_surgery" or self.nutrition_type in ["enteral", "parenteral"]:
            return "regular"  # 每6小时
        elif self.patient_type == "oral_diet":
            return "meal"  # 进食前1小时及晚12点
        else:
            return "regular"

    def _get_next_check_time(self, current_time_min):
        """获取下一次血糖检测时间（分钟）"""
        current_hour = (current_time_min / 60) % 24

        if self.monitoring_frequency == "intensive":
            next_hour = int(current_hour) + 1
            if next_hour >= 24:
                next_hour = 0
            return next_hour * 60

        elif self.monitoring_frequency == "regular":
            for check_hour in MONITORING_TIMES_REGULAR:
                if check_hour > current_hour:
                    return check_hour * 60
            return MONITORING_TIMES_REGULAR[0] * 60 + 24 * 60  # 下一天的第一个检查时间

        elif self.monitoring_frequency == "meal":
            for check_hour in MONITORING_TIMES_MEAL:
                if check_hour > current_hour:
                    return check_hour * 60
            return MONITORING_TIMES_MEAL[0] * 60 + 24 * 60  # 下一天的第一个检查时间

        # 如果连续2天血糖低于10 mmol/L且糖量摄入无变化，监测频率可调整为每12小时一次
        elif self.monitoring_frequency == "stable":
            if current_hour < 6:
                return 6 * 60
            elif current_hour < 18:
                return 18 * 60
            else:
                return 6 * 60 + 24 * 60

    def _adjust_insulin_pump(self, bg_value):
        """根据血糖值调整胰岛素泵速率"""
        # 低血糖处理
        if bg_value < ICU_BG_CRITICAL_LOW:  # < 4.0 mmol/L
            self.insulin_pump_rate = 0
            logger.info(f"t = {self.t}, 低血糖警报! BG = {bg_value:.2f} mmol/L, 停止胰岛素, 需要50%葡萄糖溶液20ml")
            return

        if bg_value < ICU_BG_TARGET_LOW:  # < 7.8 mmol/L
            self.insulin_pump_rate = 0
            logger.info(f"t = {self.t}, 血糖低于目标范围: {bg_value:.2f} mmol/L, 停止胰岛素")
            return

        # 高血糖处理
        if not self.insulin_pump_active and bg_value > ICU_BG_TARGET_HIGH:
            # 启动胰岛素泵
            self.insulin_pump_active = True

            # 根据血糖值设置初始剂量
            if 10.0 <= bg_value <= 13.0:
                self.insulin_pump_rate = 2
            elif 13.1 <= bg_value <= 17.0:
                self.insulin_pump_rate = 4
            elif 17.1 <= bg_value <= 20.0:
                self.insulin_pump_rate = 5
            elif bg_value > 20.0:
                self.insulin_pump_rate = 6
                logger.warning(f"t = {self.t}, 严重高血糖! BG = {bg_value:.2f} mmol/L, 请通知医生")

            logger.info(f"t = {self.t}, 启动胰岛素泵: BG = {bg_value:.2f} mmol/L, 泵速 = {self.insulin_pump_rate} U/h")

        # 已经在使用胰岛素泵，需要调整剂量
        elif self.insulin_pump_active:
            if self.last_bg_value is not None:
                bg_diff = bg_value - self.last_bg_value

                # 血糖仍高且上升
                if bg_value > ICU_BG_TARGET_LOW and bg_diff >= 1.0:
                    self.insulin_pump_rate += 1
                    logger.info(
                        f"t = {self.t}, 血糖上升: {bg_value:.2f} mmol/L (+{bg_diff:.2f}), 增加胰岛素至 {self.insulin_pump_rate} U/h")

                # 血糖下降过快
                elif bg_diff <= -1.0:
                    self.insulin_pump_rate = max(0, self.insulin_pump_rate - 1)
                    logger.info(
                        f"t = {self.t}, 血糖下降: {bg_value:.2f} mmol/L ({bg_diff:.2f}), 减少胰岛素至 {self.insulin_pump_rate} U/h")

                # 血糖恢复正常后重新启动胰岛素泵
                elif not self.insulin_pump_active and bg_value > ICU_BG_TARGET_HIGH:
                    self.insulin_pump_active = True
                    self.insulin_pump_rate = max(1, self.insulin_pump_rate // 2)  # 减半
                    logger.info(
                        f"t = {self.t}, 重新启动胰岛素泵: BG = {bg_value:.2f} mmol/L, 泵速 = {self.insulin_pump_rate} U/h")

    def _check_blood_glucose(self):
        """检查血糖并根据ICU常规进行处理"""
        # 获取当前血糖值 (mg/dL转换为mmol/L)
        bg_mg_dl = self.observation.Gsub
        bg_mmol_l = bg_mg_dl / 18.0  # 转换为mmol/L

        logger.info(f"t = {self.t}, 血糖检测: {bg_mmol_l:.2f} mmol/L ({bg_mg_dl:.2f} mg/dL)")

        # 调整胰岛素泵
        self._adjust_insulin_pump(bg_mmol_l)

        # 更新血糖监测相关参数
        self.last_check_time = self.t
        self.next_check_time = self.t + (self._get_next_check_time(self.t) - (self.t % (24 * 60)))
        self.last_bg_value = bg_mmol_l

        # 检查血糖稳定性，更新监测频率
        if bg_mmol_l < ICU_BG_TARGET_HIGH:
            self.stable_bg_days += 1 / 4  # 假设每天检查4次
            if self.stable_bg_days >= 2 and self.monitoring_frequency != "stable":
                self.monitoring_frequency = "stable"
                logger.info(f"t = {self.t}, 血糖连续2天稳定，调整监测频率为每12小时一次")
        else:
            self.stable_bg_days = 0
            if self.monitoring_frequency == "stable":
                self.monitoring_frequency = self._determine_monitoring_frequency()
                logger.info(f"t = {self.t}, 血糖不稳定，恢复正常监测频率")

    @classmethod
    def withID(cls, patient_id, **kwargs):
        """
        Construct patient by patient_id
        id are integers from 1 to 30.
        1  - 10: adolescent#001 - adolescent#010
        11 - 20: adult#001 - adult#001
        21 - 30: child#001 - child#010
        """
        patient_params = pd.read_csv(PATIENT_PARA_FILE)
        params = patient_params.iloc[patient_id - 1, :]
        return cls(params, **kwargs)

    @classmethod
    def withName(cls, name, **kwargs):
        """
        Construct patient by name.
        Names can be
            adolescent#001 - adolescent#010
            adult#001 - adult#001
            child#001 - child#010
        """
        patient_params = pd.read_csv(PATIENT_PARA_FILE)
        params = patient_params.loc[patient_params.Name == name].squeeze()
        return cls(params, **kwargs)

    @property
    def state(self):
        return self._odesolver.y

    @property
    def t(self):
        return self._odesolver.t

    @property
    def sample_time(self):
        return self.SAMPLE_TIME

    def step(self, action):
        # 检查是否需要进行血糖监测
        if self.next_check_time is not None and self.t >= self.next_check_time:
            self._check_blood_glucose()

        # 如果胰岛素泵活跃，使用泵速率
        if self.insulin_pump_active:
            action = action._replace(insulin=self.insulin_pump_rate / 60)  # 转换为U/min

        # 原有的进餐处理逻辑
        to_eat = self._announce_meal(action.CHO)
        action = action._replace(CHO=to_eat)

        # Detect eating or not and update last digestion amount
        if action.CHO > 0 and self._last_action.CHO <= 0:
            logger.info("t = {}, patient starts eating ...".format(self.t))
            self._last_Qsto = self.state[0] + self.state[1]  # unit: mg
            self._last_foodtaken = 0  # unit: g
            self.is_eating = True

        if to_eat > 0:
            logger.debug("t = {}, patient eats {} g".format(self.t, action.CHO))

        if self.is_eating:
            self._last_foodtaken += action.CHO  # g

        # Detect eating ended
        if action.CHO <= 0 and self._last_action.CHO > 0:
            logger.info("t = {}, Patient finishes eating!".format(self.t))
            self.is_eating = False

        # Update last input
        self._last_action = action

        # ODE solver
        self._odesolver.set_f_params(
            action, self._params, self._last_Qsto, self._last_foodtaken
        )
        if self._odesolver.successful():
            self._odesolver.integrate(self._odesolver.t + self.sample_time)
        else:
            logger.error("ODE solver failed!!")
            raise

    @staticmethod
    def model(t, x, action, params, last_Qsto, last_foodtaken):
        dxdt = np.zeros(13)
        d = action.CHO * 1000  # g -> mg
        insulin = action.insulin * 6000 / params.BW  # U/min -> pmol/kg/min
        basal = params.u2ss * params.BW / 6000  # U/min

        # Glucose in the stomach
        qsto = x[0] + x[1]
        # NOTE: Dbar is in unit mg, hence last_foodtaken needs to be converted
        # from mg to g. See https://github.com/jxx123/simglucose/issues/41 for
        # details.
        Dbar = last_Qsto + last_foodtaken * 1000  # unit: mg

        # Stomach solid
        dxdt[0] = -params.kmax * x[0] + d

        if Dbar > 0:
            aa = 5 / (2 * Dbar * (1 - params.b))
            cc = 5 / (2 * Dbar * params.d)
            kgut = params.kmin + (params.kmax - params.kmin) / 2 * (
                    np.tanh(aa * (qsto - params.b * Dbar))
                    - np.tanh(cc * (qsto - params.d * Dbar))
                    + 2
            )
        else:
            kgut = params.kmax

        # stomach liquid
        dxdt[1] = params.kmax * x[0] - x[1] * kgut

        # intestine
        dxdt[2] = kgut * x[1] - params.kabs * x[2]

        # Rate of appearance
        Rat = params.f * params.kabs * x[2] / params.BW
        # Glucose Production
        EGPt = params.kp1 - params.kp2 * x[3] - params.kp3 * x[8]
        # Glucose Utilization
        Uiit = params.Fsnc

        # renal excretion
        if x[3] > params.ke2:
            Et = params.ke1 * (x[3] - params.ke2)
        else:
            Et = 0

        # glucose kinetics
        # plus dextrose IV injection input u[2] if needed
        dxdt[3] = max(EGPt, 0) + Rat - Uiit - Et - params.k1 * x[3] + params.k2 * x[4]
        dxdt[3] = (x[3] >= 0) * dxdt[3]

        Vmt = params.Vm0 + params.Vmx * x[6]
        Kmt = params.Km0
        Uidt = Vmt * x[4] / (Kmt + x[4])
        dxdt[4] = -Uidt + params.k1 * x[3] - params.k2 * x[4]
        dxdt[4] = (x[4] >= 0) * dxdt[4]

        # insulin kinetics
        dxdt[5] = (
                -(params.m2 + params.m4) * x[5]
                + params.m1 * x[9]
                + params.ka1 * x[10]
                + params.ka2 * x[11]
        )  # plus insulin IV injection u[3] if needed
        It = x[5] / params.Vi
        dxdt[5] = (x[5] >= 0) * dxdt[5]

        # insulin action on glucose utilization
        dxdt[6] = -params.p2u * x[6] + params.p2u * (It - params.Ib)

        # insulin action on production
        dxdt[7] = -params.ki * (x[7] - It)

        dxdt[8] = -params.ki * (x[8] - x[7])

        # insulin in the liver (pmol/kg)
        dxdt[9] = -(params.m1 + params.m30) * x[9] + params.m2 * x[5]
        dxdt[9] = (x[9] >= 0) * dxdt[9]

        # subcutaneous insulin kinetics
        dxdt[10] = insulin - (params.ka1 + params.kd) * x[10]
        dxdt[10] = (x[10] >= 0) * dxdt[10]

        dxdt[11] = params.kd * x[10] - params.ka2 * x[11]
        dxdt[11] = (x[11] >= 0) * dxdt[11]

        # subcutaneous glucose
        dxdt[12] = -params.ksc * x[12] + params.ksc * x[3]
        dxdt[12] = (x[12] >= 0) * dxdt[12]

        if action.insulin > basal:
            logger.debug("t = {}, injecting insulin: {}".format(t, action.insulin))

        return dxdt

    @property
    def observation(self):
        """
        return the observation from patient
        for now, only the subcutaneous glucose level is returned
        add heart rate as an observation
        """
        GM = self.state[12]  # subcutaneous glucose (mg/kg)
        Gsub = GM / self._params.Vg
        observation = Observation(
            Gsub=Gsub,
            last_check_time=self.last_check_time,
            next_check_time=self.next_check_time
        )
        return observation

    def _announce_meal(self, meal):
        """
        patient announces meal.
        The announced meal will be added to self.planned_meal
        The meal is consumed in self.EAT_RATE
        The function will return the amount to eat at current time
        """
        self.planned_meal += meal
        if self.planned_meal > 0:
            to_eat = min(self.EAT_RATE, self.planned_meal)
            self.planned_meal -= to_eat
            self.planned_meal = max(0, self.planned_meal)
        else:
            to_eat = 0
        return to_eat

    @property
    def seed(self):
        return self._seed

    @seed.setter
    def seed(self, seed):
        self._seed = seed
        self.reset()

    def reset(self):
        """
        Reset the patient state to default intial state
        """
        if self._init_state is None:
            self.init_state = np.copy(self._params.iloc[2:15].values)
        else:
            self.init_state = self._init_state

        self.random_state = np.random.RandomState(self.seed)
        if self.random_init_bg:
            # Only randomize glucose related states, x4, x5, and x13
            mean = [
                1.0 * self.init_state[3],
                1.0 * self.init_state[4],
                1.0 * self.init_state[12],
            ]
            cov = np.diag(
                [
                    0.1 * self.init_state[3],
                    0.1 * self.init_state[4],
                    0.1 * self.init_state[12],
                ]
            )
            bg_init = self.random_state.multivariate_normal(mean, cov)
            self.init_state[3] = 1.0 * bg_init[0]
            self.init_state[4] = 1.0 * bg_init[1]
            self.init_state[12] = 1.0 * bg_init[2]

        self._last_Qsto = self.init_state[0] + self.init_state[1]
        self._last_foodtaken = 0
        self.name = self._params.Name

        self._odesolver = ode(self.model).set_integrator("dopri5")
        self._odesolver.set_initial_value(self.init_state, self.t0)

        self._last_action = Action(CHO=0, insulin=0)
        self.is_eating = False
        self.planned_meal = 0

        # 重置ICU相关参数
        self.last_check_time = None
        self.next_check_time = self.t0 + self._get_next_check_time(self.t0)
        self.insulin_pump_active = False
        self.insulin_pump_rate = 0
        self.last_bg_value = None
        self.stable_bg_days = 0


if __name__ == "__main__":
    logger.setLevel(logging.INFO)
    # create console handler and set level to debug
    ch = logging.StreamHandler()
    # ch.setLevel(logging.DEBUG)
    ch.setLevel(logging.INFO)
    # create formatter
    formatter = logging.Formatter("%(name)s: %(levelname)s: %(message)s")
    # add formatter to ch
    ch.setFormatter(formatter)
    # add ch to logger
    logger.addHandler(ch)

    # 创建ICU患者
    p = ICUPatient.withName("adolescent#001", patient_type="post_surgery", nutrition_type="fasting")
    basal = p._params.u2ss * p._params.BW / 6000  # U/min

    t = []
    CHO = []
    insulin = []
    BG = []
    BG_mmol = []
    insulin_rates = []

    # 模拟48小时
    while p.t < 2880:  # 48小时 = 2880分钟
        ins = basal
        carb = 0

        # 模拟在t=100时进食
        if p.t == 100:
            carb = 80
            logger.info(f"t = {p.t}, 患者进食 {carb}g 碳水化合物")

        # 模拟在t=1000时进食
        if p.t == 1000:
            carb = 100
            logger.info(f"t = {p.t}, 患者进食 {carb}g 碳水化合物")

        # 创建动作
        act = Action(insulin=ins, CHO=carb)

        # 记录数据
        t.append(p.t)
        CHO.append(act.CHO)
        insulin.append(act.insulin)
        BG.append(p.observation.Gsub)
        BG_mmol.append(p.observation.Gsub / 18.0)  # 转换为mmol/L
        insulin_rates.append(p.insulin_pump_rate if p.insulin_pump_active else 0)

        # 执行步骤
        p.step(act)

    import matplotlib.pyplot as plt

    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    fig, ax = plt.subplots(4, sharex=True, figsize=(12, 10))

    # 绘制血糖曲线 (mg/dL)
    ax[0].plot(t, BG, 'b-', label='血糖 (mg/dL)')
    ax[0].axhline(y=140, color='g', linestyle='--', label='目标下限 (140 mg/dL)')
    ax[0].axhline(y=180, color='r', linestyle='--', label='目标上限 (180 mg/dL)')
    ax[0].set_ylabel('血糖 (mg/dL)')
    ax[0].legend()
    ax[0].grid(True)

    # 绘制血糖曲线 (mmol/L)
    ax[1].plot(t, BG_mmol, 'b-', label='血糖 (mmol/L)')
    ax[1].axhline(y=7.8, color='g', linestyle='--', label='目标下限 (7.8 mmol/L)')
    ax[1].axhline(y=10.0, color='r', linestyle='--', label='目标上限 (10.0 mmol/L)')
    ax[1].set_ylabel('血糖 (mmol/L)')
    ax[1].legend()
    ax[1].grid(True)

    # 绘制碳水摄入
    ax[2].bar(t, CHO, color='orange', label='碳水摄入 (g)')
    ax[2].set_ylabel('碳水摄入 (g)')
    ax[2].legend()

    # 绘制胰岛素泵速率
    ax[3].plot(t, insulin_rates, 'r-', label='胰岛素泵速率 (U/h)')
    ax[3].set_ylabel('胰岛素泵速率 (U/h)')
    ax[3].set_xlabel('时间 (分钟)')
    ax[3].legend()
    ax[3].grid(True)

    plt.suptitle('ICU患者血糖控制模拟', fontsize=16)
    plt.tight_layout()
    plt.savefig('icu_glucose_control.png', dpi=300)
    plt.show()