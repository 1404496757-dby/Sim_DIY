import numpy as np
from .base import Controller, Action


class Cloud:
    """单个数据云类"""

    def __init__(self, x, time):
        self.mean = x.copy()
        self.sigma = np.dot(x, x)
        self.count = 1
        self.params = np.array([0.01, 0.001, 0.0, 0.0])  # P, I, D, R
        self.added_time = time

    def update(self, x):
        """更新云参数"""
        self.mean = (self.count - 1) / self.count * self.mean + x / self.count
        self.sigma = (self.count - 1) / self.count * self.sigma + np.dot(x, x) / self.count
        self.count += 1


class CloudManager:
    """数据云管理器"""

    def __init__(self, reset_on_sim=True):
        self.clouds = []
        self.reset_on_sim = reset_on_sim  # 是否在仿真时重置

    def reset(self):
        """重置云数据"""
        if self.reset_on_sim:
            self.clouds = []

    def add_cloud(self, x, time):
        """添加新云"""
        new_cloud = Cloud(x, time)
        self.clouds.append(new_cloud)
        return new_cloud

    def get_active_cloud(self, x):
        """获取活跃云及最大密度"""
        if not self.clouds:
            return None, 0

        densities = [self._local_density(x, cloud) for cloud in self.clouds]
        max_idx = np.argmax(densities)
        return self.clouds[max_idx], densities[max_idx]

    def _local_density(self, x, cloud):
        """计算局部密度"""
        diff = x - cloud.mean
        return 1 / (1 + np.dot(diff, diff) + cloud.sigma - np.dot(cloud.mean, cloud.mean))

    def initialize_params(self, new_cloud):
        """初始化新云参数"""
        if self.clouds:
            weights = [c.count for c in self.clouds]
            total = sum(weights)
            new_cloud.params = sum([c.params * w for c, w in zip(self.clouds, weights)]) / total


class RECCoGlucoseController(Controller):
    def __init__(self, target=120, safe_min=70, safe_max=180, Ts=3, tau=60,
                 y_range=(60, 150), reset_clouds=False):
        """
        改进版RECCo控制器

        参数：
        reset_clouds: 是否每次仿真重置数据云 (默认False)
        """
        # 初始化控制参数
        self.target = target
        self.safe_min = safe_min
        self.safe_max = safe_max
        self.u_range = (0, 5)
        self.bolus_range = (0, 10)
        self.Ts = Ts
        self.tau = tau
        self.y_range = y_range
        self.Delta_y = y_range[1] - y_range[0]
        self.Delta_e = self.Delta_y / 2

        # 初始化云管理器
        self.cloud_manager = CloudManager(reset_on_sim=reset_clouds)

        # 其他参数
        self._init_controller_params()

    def _init_controller_params(self):
        """初始化非云相关参数"""
        self.a_r = 1 - self.Ts / 60 / self.tau
        self.alpha = 0.1 * (self.u_range[1] - self.u_range[0]) / 20
        self.G_sign = -1
        self.n_add = 12
        self.d_dead = 5
        self.sigma_L = 1e-6
        self.time = 0
        self.Sigma_e = 0
        self.last_e = 0
        self.last_CGM = None
        self.last_action = Action(basal=0, bolus=0)

    def policy(self, observation, reward, done, **info):
        """控制策略"""
        CGM = observation.CGM
        self.time += 1

        # 1. 参考模型
        y_ref = self.a_r * getattr(self, 'y_ref_prev', self.target) + (1 - self.a_r) * self.target
        self.y_ref_prev = y_ref

        # 2. 误差计算
        e = y_ref - CGM
        x = np.array([e / self.Delta_e, (y_ref - self.y_range[0]) / self.Delta_y])

        # 3. 云演化
        active_cloud, max_density = self.cloud_manager.get_active_cloud(x)

        # 添加新云条件
        if (max_density < 0.93 and
                (self.time - getattr(active_cloud, 'added_time', -np.inf)) > self.n_add):
            new_cloud = self.cloud_manager.add_cloud(x, self.time)
            self.cloud_manager.initialize_params(new_cloud)
            active_cloud = new_cloud

        # 4. 控制计算
        basal = bolus = 0
        if active_cloud:
            # ... (保持原有控制逻辑)
            pass

        return Action(basal=basal, bolus=bolus)

    def reset(self):
        """重置控制器"""
        self._init_controller_params()
        self.cloud_manager.reset()  # 根据reset_clouds参数决定是否重置云

'''
import numpy as np
from collections import deque


class RECCoController:
    def __init__(self, u_range, y_range, tau, Ts, G_sign=1):
        """
        初始化RECCo控制器

        参数：
        u_range: (u_min, u_max) 控制输入范围
        y_range: (y_min, y_max) 输出范围
        tau: 估计的时间常数
        Ts: 采样时间
        G_sign: 过程增益符号（+1/-1）
        """
        # 系统参数
        self.u_min, self.u_max = u_range
        self.y_min, self.y_max = y_range
        self.Delta_y = y_range[1] - y_range[0]
        self.Delta_e = self.Delta_y / 2
        self.Ts = Ts
        self.tau = tau

        # 参考模型参数
        self.a_r = 1 - Ts / tau

        # 自适应参数
        self.alpha = 0.1 * (u_range[1] - u_range[0]) / 20  # 缩放后的基础学习率
        self.G_sign = G_sign
        self.n_add = 20  # 添加新云的最小间隔
        self.last_add_time = -np.inf

        # 数据云存储结构 [dict]
        self.clouds = []  # 每个元素为{
        #   'mean': 均值,
        #   'sigma': 均方长度,
        #   'count': 数据点数,
        #   'params': [P, I, D, R],
        #   'added_time': 添加时间
        # }

        # 状态变量
        self.Sigma_e = 0  # 误差积分
        self.last_e = 0  # 上一次误差
        self.last_u = 0  # 上一次控制量
        self.time = 0  # 当前时间步

        # 鲁棒性参数
        self.d_dead = 0.05 * self.Delta_y  # 死区阈值
        self.sigma_L = 1e-6  # 泄漏系数

    def _normalize(self, e, y_ref):
        """数据标准化（公式4）"""
        return np.array([
            e / self.Delta_e,
            (y_ref - self.y_min) / self.Delta_y
        ])

    def _local_density(self, x, cloud):
        """计算局部密度（公式6）"""
        diff = x - cloud['mean']
        return 1 / (1 + np.dot(diff, diff) + cloud['sigma'] - np.dot(cloud['mean'], cloud['mean']))

    def _update_cloud(self, cloud, x):
        """更新云参数（公式7-8）"""
        M = cloud['count']
        cloud['mean'] = (M - 1) / M * cloud['mean'] + x / M
        cloud['sigma'] = (M - 1) / M * cloud['sigma'] + np.dot(x, x) / M
        cloud['count'] += 1
        return cloud

    def _add_cloud(self, x):
        """添加新云并初始化参数"""
        new_cloud = {
            'mean': x.copy(),
            'sigma': np.dot(x, x),
            'count': 1,
            'params': self._initialize_params(),
            'added_time': self.time
        }
        self.clouds.append(new_cloud)
        return new_cloud

    def _initialize_params(self):
        """初始化新云的PID-R参数"""
        if not self.clouds:  # 第一个云
            return np.zeros(4)

        # 加权平均现有参数（公式12）
        weights = [c['count'] for c in self.clouds]
        total = sum(weights)
        return sum([c['params'] * w for c, w in zip(self.clouds, weights)]) / total

    def reference_model(self, r):
        """参考模型（公式1）"""
        if not hasattr(self, 'y_ref_prev'):
            self.y_ref_prev = r
        y_ref = self.a_r * self.y_ref_prev + (1 - self.a_r) * r
        self.y_ref_prev = y_ref
        return y_ref

    def control_step(self, y_meas, r):
        """执行控制计算"""
        self.time += 1

        # 1. 参考模型
        y_ref = self.reference_model(r)
        e = y_ref - y_meas

        # 2. 数据预处理
        x = self._normalize(e, y_ref)
        normalized_error = e / self.Delta_e

        # 3. 演化机制
        densities=[]
        if self.clouds:
            densities = [self._local_density(x, c) for c in self.clouds]
            max_density = max(densities)
            active_idx = np.argmax(densities)
        else:
            max_density = 0
            active_idx = -1

        # 判断是否需要添加新云
        if (max_density < 0.93 and
                (self.time - self.last_add_time) > self.n_add):
            new_cloud = self._add_cloud(x)
            active_idx = len(self.clouds) - 1
            self.last_add_time = self.time
            densities.append(0.93)  # 保证新云被选中

        # 4. 参数自适应
        if self.clouds:
            active_cloud = self.clouds[active_idx]
            params = active_cloud['params']
            P, I, D, R = params

            # 计算积分项和微分项
            Delta_e = e - self.last_e
            if self.last_u < self.u_max and self.last_u > self.u_min:
                self.Sigma_e += e

            # 自适应律（公式14）
            if abs(e) > self.d_dead:  # 死区
                denom = 1 + r ** 2
                delta_P = self.alpha * self.G_sign * (abs(e * normalized_error)) / denom
                delta_I = self.alpha * self.G_sign * (abs(e * self.Sigma_e)) / denom
                delta_D = self.alpha * self.G_sign * (abs(e * Delta_e)) / denom
                delta_R = self.alpha * self.G_sign * e / denom

                # 泄漏（公式19）
                params = (1 - self.sigma_L) * params + [delta_P, delta_I, delta_D, delta_R]

                # 参数投影（公式18）
                params[:3] = np.maximum(params[:3], 0)  # P,I,D >=0
                active_cloud['params'] = params

                # 5. 计算控制量
        if self.clouds:
            # 计算各云贡献（公式16）
            contributions = []
            weights = []
            for cloud in self.clouds:
                P, I, D, R = cloud['params']
                u_i = P * e + I * self.Sigma_e + D * Delta_e + R
                contributions.append(u_i)
                weights.append(self._local_density(x, cloud))

                # 加权平均
            total_weight = sum(weights)
            u = sum([u * w for u, w in zip(contributions, weights)]) / total_weight
        else:
            u = 0  # 初始状态

                # 应用输入约束
        u_clamped = np.clip(u, self.u_min, self.u_max)

                # 更新状态
        self.last_e = e
        self.last_u = u_clamped

        return u_clamped


# 使用示例
if __name__ == "__main__":
    # 初始化控制器参数（示例值）
    controller = RECCoController(
        u_range=(0, 5),
        y_range=(0, 10),
        tau=40,
        Ts=1,
        G_sign=1
    )

    # 模拟控制循环
    y_process = 0
    for t in range(1000):
        r = 5 if t < 500 else 3  # 参考信号
        y_meas = y_process + np.random.normal(0, 0.005)  # 添加测量噪声

        u = controller.control_step(y_meas, r)

        # 模拟过程动态（示例的一阶系统）
        y_process = 0.98 * y_process + 0.01 * u

        # 记录数据用于可视化
        print(f"Time: {t}, Ref: {r:.2f}, Output: {y_meas:.2f}, Control: {u:.2f}")
'''