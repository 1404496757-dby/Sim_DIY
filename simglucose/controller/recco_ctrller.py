from .base import Controller
from .base import Action
import numpy as np
import matplotlib.pyplot as plt
import logging

logger = logging.getLogger(__name__)


class PIDController(Controller):
    def __init__(self, process_params, evolving_params, adaptation_params,target=140):
        self.target = target
        # 过程参数
        self.Ts = process_params['Ts']
        self.u_min = process_params['u_min']
        self.u_max = process_params['u_max']
        self.r_min = process_params['r_min']
        self.r_max = process_params['r_max']
        self.tau = process_params['tau']

        # 计算归一化参数
        self.Delta_r = self.r_max - self.r_min
        self.Delta_epsilon = self.Delta_r / 2

        # 进化参数
        self.gamma_max = evolving_params.get('gamma_max', 0.93)
        self.n_add = evolving_params.get('n_add', 20)
        self.max_rules = evolving_params.get('max_rules', 9997)

        # 自适应参数
        self.alpha_P = adaptation_params['alpha_P']
        self.alpha_I = adaptation_params['alpha_I']
        self.alpha_D = adaptation_params['alpha_D']
        self.alpha_R = adaptation_params['alpha_R']
        self.G_sign = adaptation_params['G_sign']
        self.d_dead = adaptation_params['d_dead']
        self.sigma_L = adaptation_params.get('sigma_L', 1e-6)

        # 初始化变量
        self.rules = []  # 存储所有规则(云)的列表
        self.last_add_time = -self.n_add  # 上次新增规则的时间

        # 参考模型参数
        self.a_r = 1 - self.Ts / self.tau

        # 控制变量
        self.y_r_prev = 0  # 上一时刻参考模型输出
        self.e_prev = 0  # 上一时刻跟踪误差
        self.Sigma_e = 0  # 误差积分项
        self.k = 0  # 时间步长计数器

        self.Delta_epsilon = 1  # 假设值，需根据实际初始化
        self.r_min = 0  # 假设值，需根据实际初始化
        self.Delta_r = 1  # 假设值，需根据实际初始化
        self.x1_data = []
        self.x2_data = []
        self.cloud_ids = []
        self.new_cloud_times = []

    def policy(self, observation, reward, done, **kwargs):
        sample_time = kwargs.get('sample_time')

        # BG is the only state for this PID controller
        bg = observation.CGM
        insulin = bg/10

        logger.info('Control input: {}'.format(insulin))

        # return the action
        action = Action(basal=insulin, bolus=0)
        return action

    def reference_model(self, r_k):
        """一阶参考模型生成期望轨迹"""
        y_r_k = self.a_r * self.y_r_prev + (1 - self.a_r) * r_k
        self.y_r_prev = y_r_k
        return y_r_k

    def calculate_local_density(self, x_k, rule):
        """
        计算当前数据点x_k与规则rule的局部密度
        使用Cauchy核函数公式(6)-(9)
        """
        M = rule['M']  # 该规则关联的数据点数量
        mu = rule['mu']  # 均值
        sigma = rule['sigma']  # 均方长度

        if M == 0:
            return 0

        # 公式(7): 局部密度计算
        denominator = 1 + np.linalg.norm(x_k - mu) ** 2 + sigma - np.linalg.norm(mu) ** 2
        gamma_k = 1 / denominator

        return gamma_k

    def update_rule_parameters(self, x_k, rule_index):
        """更新指定规则的均值和均方长度(公式8-9)"""
        rule = self.rules[rule_index]
        M = rule['M']

        # 公式(8): 更新均值
        new_mu = (M - 1) / M * rule['mu'] + 1 / M * x_k

        # 公式(9): 更新均方长度
        new_sigma = (M - 1) / M * rule['sigma'] + 1 / M * np.linalg.norm(x_k) ** 2

        # 更新规则
        self.rules[rule_index]['mu'] = new_mu
        self.rules[rule_index]['sigma'] = new_sigma
        self.rules[rule_index]['M'] += 1

    def add_new_rule(self, x_k):
        """添加新规则(数据云)"""
        if len(self.rules) >= self.max_rules:
            return

        # 新规则的初始参数
        new_rule = {
            'mu': x_k.copy(),  # 均值初始化为当前数据点
            'sigma': np.linalg.norm(x_k) ** 2,  # 初始均方长度
            'M': 1,  # 关联数据点计数
            'theta': np.zeros(4),  # [P, I, D, R] 初始为0
            'gamma_k': 0.0,  # 初始化局部密度
            'lambda_k': 0.0  # 初始化归一化密度
        }

        # 如果不是第一条规则，则初始化参数为现有规则的平均值(公式10)
        if len(self.rules) > 0:
            avg_theta = np.mean([r['theta'] for r in self.rules], axis=0)
            new_rule['theta'] = avg_theta

        self.rules.append(new_rule)
        self.last_add_time = self.k

        self.new_cloud_times.append(self.k)

    def adapt_parameters(self, rule_index, e_k, epsilon_k, Delta_epsilon_k, r_k):
        """自适应更新规则参数(公式12)"""
        rule = self.rules[rule_index]
        lambda_k = rule['lambda_k']  # 归一化密度

        # 计算参数变化量(带绝对值的初始阶段改进)
        if self.k * self.Ts < 5 * self.tau:  # 初始阶段(5倍时间常数)
            delta_P = self.alpha_P * self.G_sign * lambda_k * (abs(e_k * epsilon_k)) / (1 + r_k ** 2)
            delta_I = self.alpha_I * self.G_sign * lambda_k * (abs(e_k * self.Sigma_e)) / (1 + r_k ** 2)
            delta_D = self.alpha_D * self.G_sign * lambda_k * (abs(e_k * Delta_epsilon_k)) / (1 + r_k ** 2)
        else:
            delta_P = self.alpha_P * self.G_sign * lambda_k * (e_k * epsilon_k) / (1 + r_k ** 2)
            delta_I = self.alpha_I * self.G_sign * lambda_k * (e_k * self.Sigma_e) / (1 + r_k ** 2)
            delta_D = self.alpha_D * self.G_sign * lambda_k * (e_k * Delta_epsilon_k) / (1 + r_k ** 2)

        delta_R = self.alpha_R * self.G_sign * lambda_k * epsilon_k / (1 + r_k ** 2)

        # 应用泄漏(公式19)
        theta_new = (1 - self.sigma_L) * rule['theta'] + np.array([delta_P, delta_I, delta_D, delta_R])

        # 参数投影(公式18): 确保P,I,D非负
        theta_new[:3] = np.maximum(theta_new[:3], 0)

        # 更新规则参数
        self.rules[rule_index]['theta'] = theta_new

    def control_law(self, e_k, epsilon_k, Delta_epsilon_k):
        """控制律计算(公式13,16)"""
        u_total = 0
        sum_gamma = 0

        for rule in self.rules:
            P, I, D, R = rule['theta']
            lambda_k = rule['lambda_k']

            # 公式13: 局部控制量
            u_i = P * e_k + I * self.Sigma_e + D * Delta_epsilon_k + R

            # 加权累加
            u_total += lambda_k * u_i
            sum_gamma += rule['gamma_k']

        # 公式16: 加权平均并加上u_min偏移
        if sum_gamma > 0:
            u_k = self.u_min + u_total / sum_gamma
        else:
            u_k = self.u_min

        # 限制输出范围
        u_k = np.clip(u_k, self.u_min, self.u_max)

        return u_k

    def reset(self):
        self.integrated_state = 0
        self.prev_state = 0
