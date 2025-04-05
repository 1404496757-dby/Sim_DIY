import numpy as np
from .base import Controller, Action


class CloudManager:
    """数据云管理器，用于存储和管理RECCo控制器的数据云"""

    _instance = None  # 单例模式

    @classmethod
    def get_instance(cls):
        """获取CloudManager的单例实例"""
        if cls._instance is None:
            cls._instance = CloudManager()
        return cls._instance

    def __init__(self):
        """初始化云管理器"""
        self.clouds = []  # 数据云列表
        self.last_add_time = -np.inf  # 上次添加云的时间
        self.time = 0  # 当前时间步

    def add_cloud(self, x, initialize_params_func):
        """添加新数据云"""
        new_cloud = {
            'mean': x.copy(),
            'sigma': np.dot(x, x),
            'count': 1,
            'params': initialize_params_func(),
            'added_time': self.time
        }
        self.clouds.append(new_cloud)
        return new_cloud

    def get_clouds(self):
        """获取所有数据云"""
        return self.clouds

    def get_time(self):
        """获取当前时间步"""
        return self.time

    def increment_time(self):
        """时间步加1"""
        self.time += 1

    def set_last_add_time(self, time):
        """设置上次添加云的时间"""
        self.last_add_time = time

    def get_last_add_time(self):
        """获取上次添加云的时间"""
        return self.last_add_time


class RECCoGlucoseController(Controller):
    def __init__(self, target=120, safe_min=70, safe_max=180, Ts=3, tau=60, y_range=(60, 150)):
        """
        针对血糖控制的RECCo控制器

        参数：
        patient_name: 患者ID，用于获取默认参数
        """
        # super().__init__()

        # 血糖控制特定参数
        self.target = target  # 目标血糖值 (mg/dL)
        self.safe_min = safe_min  # 低血糖阈值
        self.safe_max = safe_max  # 高血糖阈值

        # 控制输入范围 (U/h)
        self.u_range = (0, 5)  # 基础率范围
        self.bolus_range = (0, 10)  # 推注剂量范围

        # 时间参数 (分钟)
        self.Ts = Ts  # simglucose默认采样时间
        self.tau = tau  # 估计的血糖响应时间常数

        # 输出范围 (mg/dL)
        self.y_range = y_range  # 合理的血糖范围

        # 获取云管理器实例
        self.cloud_manager = CloudManager.get_instance()

        # 初始化RECCo核心参数
        self._init_recco_params()

        # 状态变量
        self.last_CGM = None
        self.last_action = Action(basal=0, bolus=0)

    def _init_recco_params(self):
        """初始化RECCo核心参数"""
        # 参考模型参数
        self.a_r = 1 - self.Ts / 60 / self.tau  # 转换为小时

        # 自适应参数
        self.alpha = 0.1 * (self.u_range[1] - self.u_range[0]) / 20
        self.G_sign = -1  # 胰岛素对血糖是负影响
        self.n_add = 12  # 5分钟采样，20个样本约1.5小时

        # 误差积分
        self.Sigma_e = 0
        self.last_e = 0

        # 鲁棒性参数
        self.Delta_y = self.y_range[1] - self.y_range[0]
        self.Delta_e = self.Delta_y / 2
        self.d_dead = 5  # mg/dL死区阈值
        self.sigma_L = 1e-6

    def _normalize(self, e, y_ref):
        """数据标准化"""
        return np.array([
            e / self.Delta_e,
            (y_ref - self.y_range[0]) / self.Delta_y
        ])

    def _local_density(self, x, cloud):
        """计算局部密度"""
        diff = x - cloud['mean']
        return 1 / (1 + np.dot(diff, diff) + cloud['sigma'] - np.dot(cloud['mean'], cloud['mean']))

    def _initialize_params(self):
        """初始化PID-R参数"""
        clouds = self.cloud_manager.get_clouds()
        if not clouds:
            # 初始参数基于临床经验
            return np.array([0.01, 0.001, 0.0, 0.0])  # P, I, D, R

        # 加权平均现有参数
        weights = [c['count'] for c in clouds]
        total = sum(weights)
        return sum([c['params'] * w for c, w in zip(clouds, weights)]) / total

    def policy(self, observation, reward, done, **info):
        """
        simglucose控制器接口
        输入: observation.CGM (当前血糖值)
        输出: Action(basal, bolus)
        """
        CGM = observation.CGM
        self.cloud_manager.increment_time()

        # 1. 参考模型
        if not hasattr(self, 'y_ref_prev'):
            self.y_ref_prev = self.target
        y_ref = self.a_r * self.y_ref_prev + (1 - self.a_r) * self.target
        self.y_ref_prev = y_ref

        # 2. 计算误差
        e = y_ref - CGM

        # 3. 数据预处理
        x = self._normalize(e, y_ref)

        # 4. 演化机制
        densities = []
        clouds = self.cloud_manager.get_clouds()
        if clouds:
            densities = [self._local_density(x, c) for c in clouds]
            max_density = max(densities) if densities else 0
            active_idx = np.argmax(densities) if densities else -1
        else:
            max_density = 0
            active_idx = -1

        # 判断是否需要添加新云
        current_time = self.cloud_manager.get_time()
        last_add_time = self.cloud_manager.get_last_add_time()
        if (max_density < 0.93 and
                (current_time - last_add_time) > self.n_add):
            self.cloud_manager.add_cloud(x, self._initialize_params)
            self.cloud_manager.set_last_add_time(current_time)
            clouds = self.cloud_manager.get_clouds()
            densities = [self._local_density(x, c) for c in clouds]
            active_idx = len(clouds) - 1

        # 5. 参数自适应
        basal = bolus = 0
        if clouds:
            active_cloud = clouds[active_idx]
            P, I, D, R = active_cloud['params']

            # 计算积分和微分项
            Delta_e = e - self.last_e
            if self.last_action.basal < self.u_range[1] and self.last_action.basal > self.u_range[0]:
                self.Sigma_e += e

            # 自适应律 (仅当误差较大时)
            if abs(e) > self.d_dead:
                denom = 1 + self.target ** 2
                delta_P = self.alpha * self.G_sign * abs(e * (e / self.Delta_e)) / denom
                delta_I = self.alpha * self.G_sign * abs(e * self.Sigma_e) / denom
                delta_D = self.alpha * self.G_sign * abs(e * Delta_e) / denom
                delta_R = self.alpha * self.G_sign * e / denom

                # 应用泄漏和投影
                new_params = (1 - self.sigma_L) * np.array([P, I, D, R]) + np.array(
                    [delta_P, delta_I, delta_D, delta_R])
                new_params[:3] = np.maximum(new_params[:3], 0)  # P,I,D >=0
                active_cloud['params'] = new_params

            # 6. 计算控制量 (分开计算basal和bolus)
            # basal基于长期误差 (P和I项)
            basal = np.clip(P * e + I * self.Sigma_e + R, *self.u_range)

            # bolus基于短期变化 (D项和紧急校正)
            bolus = np.clip(D * Delta_e + (0.1 if e > 20 else 0), *self.bolus_range)

            # 低血糖保护
            if CGM < 80:
                basal = 0
                bolus = 0

        # 更新状态
        self.last_e = e
        self.last_CGM = CGM
        self.last_action = Action(basal=basal, bolus=bolus)

        return self.last_action

    def reset(self):
        """重置控制器状态，但不重置云数据"""
        # 只重置控制器内部状态，不重置云数据
        self.last_CGM = None
        self.last_action = Action(basal=0, bolus=0)
        self.Sigma_e = 0
        self.last_e = 0
        if hasattr(self, 'y_ref_prev'):
            delattr(self, 'y_ref_prev')