from simglucose.controller.base import Controller
from simglucose.controller.base import Action
import numpy as np
import logging

logger = logging.getLogger(__name__)


class RECCoController(Controller):
    def __init__(self, target=140):
        # 目标血糖值
        self.target = target

        # 过程参数
        self.process_params = {
            'Ts': 5.0,  # 采样时间5分钟
            'u_min': 0.0,  # 最小胰岛素注射量
            'u_max': 10.0,  # 最大胰岛素注射量
            'r_min': 70.0,  # 最小血糖参考值
            'r_max': 180.0,  # 最大血糖参考值
            'tau': 60.0  # 参考模型时间常数(分钟)
        }

        # 进化参数
        self.evolving_params = {
            'gamma_max': 0.93,  # 密度阈值
            'n_add': 12,  # 新增云的最小间隔(约1小时)
            'max_rules': 100  # 最大规则数
        }

        # 自适应参数
        self.adaptation_params = {
            'alpha_P': 0.05,  # PID参数自适应增益
            'alpha_I': 0.01,
            'alpha_D': 0.02,
            'alpha_R': 0.01,
            'G_sign': -1,  # 过程增益符号(负，因为胰岛素降低血糖)
            'd_dead': 5.0,  # 死区阈值5 mg/dL
            'sigma_L': 1e-6  # 泄漏系数
        }

        # 初始化RECCo控制器核心组件
        self.init_recco()

        # 存储历史数据用于可视化
        self.bg_history = []
        self.insulin_history = []
        self.reference_history = []
        self.time_history = []
        self.rule_count_history = []

    def init_recco(self):
        """初始化RECCo控制器内部变量"""
        # 计算归一化参数
        self.Delta_r = self.process_params['r_max'] - self.process_params['r_min']
        self.Delta_epsilon = self.Delta_r / 2

        # 初始化变量
        self.rules = []  # 存储所有规则(云)的列表
        self.last_add_time = -self.evolving_params['n_add']  # 上次新增规则的时间

        # 参考模型参数
        self.a_r = 1 - self.process_params['Ts'] / self.process_params['tau']

        # 控制变量
        self.y_r_prev = self.target  # 上一时刻参考模型输出，初始为目标值
        self.e_prev = 0  # 上一时刻跟踪误差
        self.Sigma_e = 0  # 误差积分项
        self.k = 0  # 时间步长计数器

        # 数据存储
        self.x1_data = []
        self.x2_data = []
        self.cloud_ids = []
        self.new_cloud_times = []

    def reference_model(self, r_k):
        """一阶参考模型生成期望轨迹"""
        y_r_k = self.a_r * self.y_r_prev + (1 - self.a_r) * r_k
        self.y_r_prev = y_r_k
        return y_r_k

    def calculate_local_density(self, x_k, rule):
        """计算当前数据点x_k与规则rule的局部密度"""
        M = rule['M']  # 该规则关联的数据点数量
        mu = rule['mu']  # 均值
        sigma = rule['sigma']  # 均方长度

        if M == 0:
            return 0

        # 局部密度计算
        denominator = 1 + np.linalg.norm(x_k - mu) ** 2 + sigma - np.linalg.norm(mu) ** 2
        gamma_k = 1 / denominator

        return gamma_k

    def update_rule_parameters(self, x_k, rule_index):
        """更新指定规则的均值和均方长度"""
        rule = self.rules[rule_index]
        M = rule['M']

        # 更新均值
        new_mu = (M - 1) / M * rule['mu'] + 1 / M * x_k

        # 更新均方长度
        new_sigma = (M - 1) / M * rule['sigma'] + 1 / M * np.linalg.norm(x_k) ** 2

        # 更新规则
        self.rules[rule_index]['mu'] = new_mu
        self.rules[rule_index]['sigma'] = new_sigma
        self.rules[rule_index]['M'] += 1

    def add_new_rule(self, x_k):
        """添加新规则(数据云)"""
        if len(self.rules) >= self.evolving_params['max_rules']:
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

        # 如果不是第一条规则，则初始化参数为现有规则的平均值
        if len(self.rules) > 0:
            avg_theta = np.mean([r['theta'] for r in self.rules], axis=0)
            new_rule['theta'] = avg_theta
        else:
            # 第一条规则设置初始PID参数
            new_rule['theta'] = np.array([0.05, 0.001, 0.01, 0])

        self.rules.append(new_rule)
        self.last_add_time = self.k

        self.new_cloud_times.append(self.k)
        logger.info(f'添加新规则，当前规则数: {len(self.rules)}')

    def adapt_parameters(self, rule_index, e_k, epsilon_k, Delta_epsilon_k, r_k):
        """自适应更新规则参数"""
        rule = self.rules[rule_index]
        lambda_k = rule['lambda_k']  # 归一化密度

        # 计算参数变化量(带绝对值的初始阶段改进)
        if self.k * self.process_params['Ts'] < 5 * self.process_params['tau']:  # 初始阶段
            delta_P = self.adaptation_params['alpha_P'] * self.adaptation_params['G_sign'] * lambda_k * (
                abs(e_k * epsilon_k)) / (1 + r_k ** 2)
            delta_I = self.adaptation_params['alpha_I'] * self.adaptation_params['G_sign'] * lambda_k * (
                abs(e_k * self.Sigma_e)) / (1 + r_k ** 2)
            delta_D = self.adaptation_params['alpha_D'] * self.adaptation_params['G_sign'] * lambda_k * (
                abs(e_k * Delta_epsilon_k)) / (1 + r_k ** 2)
        else:
            delta_P = self.adaptation_params['alpha_P'] * self.adaptation_params['G_sign'] * lambda_k * (
                        e_k * epsilon_k) / (1 + r_k ** 2)
            delta_I = self.adaptation_params['alpha_I'] * self.adaptation_params['G_sign'] * lambda_k * (
                        e_k * self.Sigma_e) / (1 + r_k ** 2)
            delta_D = self.adaptation_params['alpha_D'] * self.adaptation_params['G_sign'] * lambda_k * (
                        e_k * Delta_epsilon_k) / (1 + r_k ** 2)

        delta_R = self.adaptation_params['alpha_R'] * self.adaptation_params['G_sign'] * lambda_k * epsilon_k / (
                    1 + r_k ** 2)

        # 应用泄漏
        theta_new = (1 - self.adaptation_params['sigma_L']) * rule['theta'] + np.array(
            [delta_P, delta_I, delta_D, delta_R])

        # 参数投影: 确保P,I,D非负
        theta_new[:3] = np.maximum(theta_new[:3], 0)

        # 更新规则参数
        self.rules[rule_index]['theta'] = theta_new

    def control_law(self, e_k, epsilon_k, Delta_epsilon_k):
        """控制律计算"""
        u_total = 0
        sum_gamma = 0

        for rule in self.rules:
            P, I, D, R = rule['theta']
            lambda_k = rule['lambda_k']

            # 局部控制量
            u_i = P * e_k + I * self.Sigma_e + D * Delta_epsilon_k + R

            # 加权累加
            u_total += lambda_k * u_i
            sum_gamma += rule['gamma_k']

        # 加权平均并加上u_min偏移
        if sum_gamma > 0:
            u_k = self.process_params['u_min'] + u_total / sum_gamma
        else:
            u_k = self.process_params['u_min']

        # 限制输出范围
        u_k = np.clip(u_k, self.process_params['u_min'], self.process_params['u_max'])

        return u_k

    def recco_step(self, bg):
        """RECCo控制器的单步执行"""
        self.k += 1

        # 参考值始终为目标血糖值
        r_k = self.target
        y_k = bg  # 当前血糖值

        # 1. 参考模型生成期望轨迹
        y_r_k = self.reference_model(r_k)

        # 2. 计算误差信号
        epsilon_k = y_r_k - y_k  # 跟踪误差
        e_k = r_k - y_k  # 控制误差
        Delta_epsilon_k = epsilon_k - self.e_prev  # 误差差分

        # 更新积分项，添加抗积分饱和
        self.Sigma_e = np.clip(self.Sigma_e + epsilon_k, -500, 500)

        # 3. 构建归一化数据点
        x_k = np.array([
            epsilon_k / self.Delta_epsilon,
            (y_r_k - self.process_params['r_min']) / self.Delta_r
        ])
        self.x1_data.append(x_k[0])
        self.x2_data.append(x_k[1])

        # 4. 规则进化机制
        if len(self.rules) == 0:
            # 初始状态下直接添加第一个规则
            self.add_new_rule(x_k)
            self.cloud_ids.append(0)  # 第一个点属于第一个云
        else:
            # 计算当前点与所有规则的局部密度
            max_gamma = 0
            best_rule_idx = 0

            for i, rule in enumerate(self.rules):
                gamma_k = self.calculate_local_density(x_k, rule)
                rule['gamma_k'] = gamma_k

                if gamma_k > max_gamma:
                    max_gamma = gamma_k
                    best_rule_idx = i

            # 计算归一化密度
            sum_gamma = sum(rule['gamma_k'] for rule in self.rules)
            for rule in self.rules:
                rule['lambda_k'] = rule['gamma_k'] / sum_gamma if sum_gamma > 0 else 0

            # 判断是否添加新规则
            if (max_gamma < self.evolving_params['gamma_max'] and
                    self.k - self.last_add_time >= self.evolving_params['n_add'] and
                    len(self.rules) < self.evolving_params['max_rules']):
                self.add_new_rule(x_k)
                self.cloud_ids.append(len(self.rules) - 1)  # 新点属于新创建的云
            else:
                # 更新最佳匹配规则的参数
                self.update_rule_parameters(x_k, best_rule_idx)
                self.cloud_ids.append(best_rule_idx)  # 记录当前点属于哪个云

                # 死区检查
                if abs(epsilon_k) >= self.adaptation_params['d_dead']:
                    self.adapt_parameters(best_rule_idx, e_k, epsilon_k, Delta_epsilon_k, r_k)

        # 5. 计算控制量
        u_k = self.control_law(e_k, epsilon_k, Delta_epsilon_k)

        # 保存当前误差用于下一时刻差分
        self.e_prev = epsilon_k

        # 记录历史数据
        self.bg_history.append(bg)
        self.insulin_history.append(u_k)
        self.reference_history.append(y_r_k)
        self.time_history.append(self.k * self.process_params['Ts'])
        self.rule_count_history.append(len(self.rules))

        return u_k

    def policy(self, observation, reward, done, **kwargs):
        """实现Controller接口的policy方法"""
        sample_time = kwargs.get('sample_time', 5)  # 默认采样时间5分钟

        # 更新采样时间
        if self.process_params['Ts'] != sample_time:
            self.process_params['Ts'] = sample_time
            self.a_r = 1 - self.process_params['Ts'] / self.process_params['tau']

        # 获取当前血糖值
        bg = observation.CGM

        # 使用RECCo算法计算胰岛素注射量
        insulin = self.recco_step(bg)

        logger.info(f'血糖: {bg:.1f}, 胰岛素: {insulin:.3f}, 规则数: {len(self.rules)}')

        # 返回控制动作
        action = Action(basal=insulin, bolus=0)
        return action

    def reset(self):
        """重置控制器状态"""
        self.init_recco()
        self.bg_history = []
        self.insulin_history = []
        self.reference_history = []
        self.time_history = []
        self.rule_count_history = []
