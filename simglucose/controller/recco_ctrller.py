import numpy as np
import pandas as pd
import csv
import time
from .base import Controller, Action
import logging

logger = logging.getLogger(__name__)


class RECCoController(Controller):
    def __init__(self,
                 y_min=0,
                 y_max=200,
                 u_min=0,
                 u_max=10,
                 tau=40,
                 Ts=1,
                 target=140,
                 cloud_file='recco_clouds.csv'):
        self.target = target
        self.y_min = y_min
        self.y_max = y_max
        self.u_min = u_min
        self.u_max = u_max
        self.tau = tau
        self.Ts = Ts
        self.cloud_file = cloud_file
        self.ar = 1 - Ts / tau  # 参考模型极点

        # 初始化云数据
        self.clouds = []  # 每个云包含: mu, sigma, M, k_add, P, I, D, R
        self.load_clouds()

        # 适应参数
        self.alpha = 0.1  # 基础适应增益
        self.alpha_adjust = (u_max - u_min) / 20 * self.alpha
        self.g_sign = 1  # 假设过程增益为正
        self.d_dead = 0.1  # 死区阈值
        self.sigma_L = 1e-6  # 泄漏因子
        self.theta_low = np.array([0, 0, 0, -np.inf])  # 参数下限 (P,I,D,R)
        self.theta_high = np.array([np.inf, np.inf, np.inf, np.inf])  # 参数上限

        # 控制状态
        self.y_r_prev = target
        self.epsilon_prev = 0
        self.integral = 0
        self.c = len(self.clouds)
        self.k = 0  # 时间步

    def load_clouds(self):
        """从指定路径的 CSV 加载云数据"""
        try:
            with open(self.cloud_file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    cloud = {
                        'mu': np.array([float(row['mu1']), float(row['mu2'])]),
                        'sigma': float(row['sigma']),
                        'M': int(row['M']),
                        'k_add': int(row['k_add']),
                        'theta': np.array([
                            float(row['P']),
                            float(row['I']),
                            float(row['D']),
                            float(row['R'])
                        ])
                    }
                    self.clouds.append(cloud)
            self.c = len(self.clouds)
        except FileNotFoundError:
            pass  # 初始化时文件不存在，创建空云列表

    def save_clouds(self):
        """将云数据保存到指定路径的 CSV"""
        fieldnames = ['mu1', 'mu2', 'sigma', 'M', 'k_add', 'P', 'I', 'D', 'R']
        with open(self.cloud_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for cloud in self.clouds:
                writer.writerow({
                    'mu1': cloud['mu'][0],
                    'mu2': cloud['mu'][1],
                    'sigma': cloud['sigma'],
                    'M': cloud['M'],
                    'k_add': cloud['k_add'],
                    'P': cloud['theta'][0],
                    'I': cloud['theta'][1],
                    'D': cloud['theta'][2],
                    'R': cloud['theta'][3]
                })

    def _normalize_data(self, epsilon, y_r):
        """归一化数据点"""
        delta_y = self.y_max - self.y_min
        delta_epsilon = delta_y / 2
        x1 = epsilon / delta_epsilon
        x2 = (y_r - self.y_min) / delta_y
        return np.array([x1, x2])

    def _evolve_clouds(self, x):
        """演化法则：添加新云或更新现有云"""
        if self.c == 0:
            # 初始化第一个云
            new_cloud = {
                'mu': x.copy(),
                'sigma': np.linalg.norm(x) ** 2,
                'M': 1,
                'k_add': self.k,
                'theta': np.zeros(4)  # 初始参数全零
            }
            self.clouds.append(new_cloud)
            self.c = 1
            gamma = np.array([1])
            return 0, gamma  # 新云索引和 gamma

        # 计算所有云的局部密度
        gamma = []
        for cloud in self.clouds:
            norm_sq = np.linalg.norm(x - cloud['mu']) ** 2
            gamma_i = 1 / (1 + norm_sq + cloud['sigma'] - np.linalg.norm(cloud['mu']) ** 2)
            gamma.append(gamma_i)
        gamma = np.array(gamma)
        lambda_total = gamma.sum()
        lambda_i = gamma / lambda_total if lambda_total != 0 else np.zeros(self.c)

        max_gamma = gamma.max()
        max_idx = np.argmax(gamma)

        # 检查是否添加新云
        add_condition = (max_gamma < 0.93) and (self.k > (self.clouds[max_idx]['k_add'] + 20))
        if add_condition and self.c < 20:  # 文献中 cmax=20
            # 初始化新云参数为现有云的加权平均
            theta_init = np.sum([l * cloud['theta'] for l, cloud in zip(lambda_i, self.clouds)], axis=0)
            new_cloud = {
                'mu': x.copy(),
                'sigma': np.linalg.norm(x) ** 2,
                'M': 1,
                'k_add': self.k,
                'theta': theta_init
            }
            self.clouds.append(new_cloud)
            self.c += 1
            return self.c - 1, gamma  # 新云索引和 gamma
        else:
            # 更新关联云
            cloud = self.clouds[max_idx]
            cloud['mu'] = ((cloud['M'] - 1) * cloud['mu'] + x) / cloud['M']
            cloud['sigma'] = ((cloud['M'] - 1) * cloud['sigma'] + np.linalg.norm(x) ** 2) / cloud['M']
            cloud['M'] += 1
            return max_idx, gamma  # 关联云索引和 gamma

    def _adapt_parameters(self, active_idx, epsilon, delta_epsilon, integral_epsilon, r):
        """适应法则：更新 PID - R 参数"""
        cloud = self.clouds[active_idx]
        theta_prev = cloud['theta']

        # 计算适应增量
        norm_r = 1 + r ** 2
        delta_P = self.alpha_adjust * self.g_sign * cloud['lambda'] * (np.abs(epsilon) * np.abs(epsilon)) / norm_r
        delta_I = self.alpha_adjust * self.g_sign * cloud['lambda'] * (np.abs(epsilon) * np.abs(delta_epsilon)) / norm_r
        delta_D = self.alpha_adjust * self.g_sign * cloud['lambda'] * (np.abs(epsilon) * np.abs(delta_epsilon)) / norm_r
        delta_R = self.alpha_adjust * self.g_sign * cloud['lambda'] * epsilon / norm_r
        delta_theta = np.array([delta_P, delta_I, delta_D, delta_R])

        # 死区机制
        if np.abs(epsilon) < self.d_dead:
            delta_theta = np.zeros(4)

        # 泄漏机制
        theta_candidate = (1 - self.sigma_L) * theta_prev + delta_theta

        # 参数投影
        theta_clamped = np.clip(theta_candidate, self.theta_low, self.theta_high)

        # 更新参数
        cloud['theta'] = theta_clamped

    def _compute_control(self, active_idx, epsilon, integral_epsilon, delta_epsilon):
        """计算控制信号"""
        cloud = self.clouds[active_idx]
        P, I, D, R = cloud['theta']

        # 反饱和积分
        u_prev = self.prev_u if hasattr(self, 'prev_u') else 0
        if self.u_min < u_prev < self.u_max:
            self.integral += epsilon * self.Ts
        else:
            self.integral = self.integral  # 保持上次积分值

        u_local = P * epsilon + I * self.integral + D * delta_epsilon + R

        # 加权平均多个云的输出（当前简化为单云或激活云）
        # 文献中使用加权平均，但为简化当前场景假设单激活云
        u = u_local

        # 限制输出范围
        u = np.clip(u, self.u_min, self.u_max)
        self.prev_u = u  # 保存上一时刻控制量
        return u

    def policy(self, observation, reward, done, **kwargs):
        """主控制策略，兼容 basal_bolus_ctrller 接口"""
        cgm = observation.CGM
        sample_time = kwargs.get('sample_time', 1)
        patient_name = kwargs.get('patient_name', 'adult#001')
        meal = kwargs.get('meal', 0)  # 暂时忽略 meal，专注血糖控制

        # 1. 参考模型计算
        r = self.target  # 目标血糖
        self.y_r_current = self.ar * self.y_r_prev + (1 - self.ar) * r
        epsilon = self.y_r_current - cgm
        delta_epsilon = epsilon - self.epsilon_prev

        # 2. 归一化数据点
        x = self._normalize_data(epsilon, self.y_r_current)

        # 3. 演化云
        active_idx, gamma = self._evolve_clouds(x)
        self.clouds[active_idx]['lambda'] = gamma[active_idx] / gamma.sum()  # 关联度

        # 4. 适应参数
        self._adapt_parameters(active_idx, epsilon, delta_epsilon, self.integral, r)

        # 5. 计算控制信号
        u = self._compute_control(active_idx, epsilon, self.integral, delta_epsilon)

        # 6. 保存状态
        self.y_r_prev = self.y_r_current
        self.epsilon_prev = epsilon
        self.k += 1

        # 7. 保存云数据
        if self.k % 10 == 0:  # 定期保存
            self.save_clouds()

        return Action(basal=u, bolus=0)  # 简化为总胰岛素剂量，基线 + bolus 可扩展

    def reset(self):
        """重置控制器状态"""
        self.clouds = []
        self.load_clouds()
        self.y_r_prev = self.target
        self.epsilon_prev = 0
        self.integral = 0
        self.k = 0
        self.prev_u = 0
