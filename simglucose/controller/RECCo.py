from .base import Controller, Action
import numpy as np


class RECCoController(Controller):
    def __init__(self, target_range=(7.8, 10.0), u_range=(0, 6), Ts=60, tau=3600):
        self.target_min, self.target_max = target_range  # 血糖目标范围 (mmol/L)
        self.u_min, self.u_max = u_range  # 胰岛素输注范围 (U/h)
        self.Ts = Ts  # 采样时间 (秒)
        self.tau = tau  # 时间常数 (秒)
        self.ar = 1 - Ts / tau  # 参考模型参数

        # 云数据结构
        self.clouds = []  # 每个云：[mean, mean_sq, count, params]
        self.last_update = 0

        # 跟踪误差相关
        self.error_sum = 0  # 误差积分项
        self.last_error = 0  # 上次误差
        self.ref_model_output = None  # 参考模型输出

        # 自适应参数
        self.alpha = 0.1 * (u_range[1] - u_range[0]) / 20  # 缩放后的自适应增益
        self.gamma_max = 0.93  # 云添加阈值
        self.n_add = 20  # 最小添加间隔

        # ICU应急处理状态
        self.insulin_suspended = False

    def policy(self, observation, reward, done, **info):
        bg = observation.CGM  # 当前血糖值
        current_time = info.get('time', 0)

        # 高低血糖应急处理
        if bg < 4.0:
            return self._handle_hypoglycemia()
        elif bg > 20.0:
            return Action(basal=0, bolus=0)  # 停止并通知医生

        # 初始化参考模型
        if self.ref_model_output is None:
            self.ref_model_output = bg

        # 计算参考轨迹
        target = (self.target_min + self.target_max) / 2  # 目标中值
        self.ref_model_output = self.ar * self.ref_model_output + (1 - self.ar) * target

        # 计算跟踪误差
        error = self.ref_model_output - bg
        delta_error = error - self.last_error

        # 更新误差积分（带抗饱和）
        if not self.insulin_suspended and self.u_min < self.current_insulin < self.u_max:
            self.error_sum += error
        self.last_error = error

        # 构建数据点
        x = self._normalize_data_point(error, bg)

        # 云结构演化
        self._update_cloud_structure(x, current_time)

        # 计算控制信号
        u = self._calculate_control(x, error, delta_error)

        # 应用胰岛素限制
        basal = np.clip(u, self.u_min, self.u_max)
        return Action(basal=basal, bolus=0)

    def _normalize_data_point(self, error, bg):
        delta_y = self.target_max - self.target_min
        delta_error = delta_y / 2
        return np.array([
            error / delta_error,
            (bg - self.target_min) / delta_y
        ])

    def _update_cloud_structure(self, x, current_time):
        if not self.clouds:
            # 初始化第一个云
            self.clouds.append({
                'mean': x.copy(),
                'mean_sq': np.dot(x, x),
                'count': 1,
                'params': np.zeros(4),  # [P, I, D, R]
                'last_used': current_time
            })
            return

        # 计算各云密度
        densities = []
        for cloud in self.clouds:
            diff = x - cloud['mean']
            density = 1 / (1 + np.dot(diff, diff) + cloud['mean_sq'] - np.dot(cloud['mean'], cloud['mean']))
            densities.append(density)

        max_density = max(densities)
        active_idx = densities.index(max_density)

        # 更新活跃云参数
        cloud = self.clouds[active_idx]
        cloud['count'] += 1
        cloud['mean'] = (cloud['count'] - 1) / cloud['count'] * cloud['mean'] + x / cloud['count']
        cloud['mean_sq'] = (cloud['count'] - 1) / cloud['count'] * cloud['mean_sq'] + np.dot(x, x) / cloud['count']
        cloud['last_used'] = current_time

        # 添加新云条件
        if (max_density < self.gamma_max and
                current_time - self.last_update > self.n_add):
            new_params = sum(c['params'] * d for c, d in zip(self.clouds, densities)) / sum(densities)
            self.clouds.append({
                'mean': x.copy(),
                'mean_sq': np.dot(x, x),
                'count': 1,
                'params': new_params,
                'last_used': current_time
            })
            self.last_update = current_time

    def _calculate_control(self, x, error, delta_error):
        if not self.clouds:
            return 0

        # 计算各云权重
        densities = []
        for cloud in self.clouds:
            diff = x - cloud['mean']
            density = 1 / (1 + np.dot(diff, diff) + cloud['mean_sq'] - np.dot(cloud['mean'], cloud['mean']))
            densities.append(density)
        weights = np.array(densities) / sum(densities)
        # 自适应参数更新
        for i, cloud in enumerate(self.clouds):
            if weights[i] < 0.1:  # 只更新活跃云
                continue

            # 带死区的参数更新
            if abs(error) > 1.0:  # 死区阈值
                delta_p = self.alpha * np.sign(1) * weights[i] * abs(error * error) / (1 + error ** 2)
                delta_i = self.alpha * np.sign(1) * weights[i] * abs(error * delta_error) / (1 + error ** 2)
                delta_r = self.alpha * np.sign(1) * weights[i] * error / (1 + error ** 2)

                cloud['params'][0] += delta_p  # P
                cloud['params'][1] += delta_i  # I
                cloud['params'][3] += delta_r  # R

        # 计算各云控制量
        contributions = []
        for cloud in self.clouds:
            P, I, D, R = cloud['params']
            u_cloud = P * error + I * self.error_sum + D * delta_error + R
            contributions.append(u_cloud)

        # 加权平均
        return np.dot(weights, contributions)

    def _handle_hypoglycemia(self):
        self.insulin_suspended = True
        self.error_sum = 0  # 重置积分项
        return Action(basal=0, bolus=0)  # 停止胰岛素

    def reset(self):
        self.clouds = []
        self.error_sum = 0
        self.last_error = 0
        self.ref_model_output = None
        self.insulin_suspended = False
