from simglucose.controller.base import Controller
from simglucose.controller.base import Action
import numpy as np
import logging
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import threading
import time
from collections import deque

logger = logging.getLogger(__name__)


class RECCoController(Controller):
    def __init__(self, target=140):
        # 目标血糖值
        self.target = target

        # 过程参数
        self.process_params = {
            'Ts': 2.0,  # 采样时间5分钟
            'u_min': 0.0,  # 最小胰岛素注射量
            'u_max': 20.0,  # 最大胰岛素注射量
            'r_min': 70.0,  # 最小血糖参考值
            'r_max': 180.0,  # 最大血糖参考值
            'tau': 40.0  # 参考模型时间常数(分钟)
        }

        # 进化参数
        self.evolving_params = {
            'gamma_max': 0.93,  # 密度阈值
            'n_add': 20,  # 新增云的最小间隔(约1小时)
            'max_rules': 100  # 最大规则数
        }

        # 自适应参数
        self.adaptation_params = {
            'alpha_P': 0.1,  # PID参数自适应增益
            'alpha_I': 0.1,
            'alpha_D': 0.1,
            'alpha_R': 0.1,
            'G_sign': -1,  # 过程增益符号(负，因为胰岛素降低血糖)
            'd_dead': 5.0,  # 死区阈值5 mg/dL
            'sigma_L': 1e-6  # 泄漏系数
        }

        self.Ts = self.evolving_params['Ts']
        self.u_min = self.evolving_params['u_min']
        self.u_max = self.evolving_params['u_max']
        self.r_min = self.evolving_params['r_min']
        self.r_max = self.evolving_params['r_max']
        self.tau = self.evolving_params['tau']

        self.gamma_max = self.evolving_params['gamma_max']
        self.max_rules = self.evolving_params['max_rules']
        self.n_add = self.evolving_params['n_add']

        self.alpha_P = self.evolving_params['alpha_P']
        self.alpha_I = self.evolving_params['alpha_I']
        self.alpha_D = self.evolving_params['alpha_D']
        self.alpha_R = self.evolving_params['alpha_R']
        self.G_sign = self.evolving_params['G_sign']
        self.d_dead = self.evolving_params['d_dead']
        self.sigma_L = self.evolving_params['sigma_L']

        # 初始化RECCo控制器核心组件
        self.init_recco()

        # 存储历史数据用于可视化
        self.bg_history = []
        self.insulin_history = []
        self.reference_history = []
        self.time_history = []
        self.rule_count_history = []

        # 实时绘图相关变量
        self.plot_lock = threading.Lock()  # 数据访问锁
        self.plot_thread = None  # 绘图线程
        self.plot_running = False  # 绘图线程运行标志
        self.plot_initialized = False  # 绘图初始化标志

        # 用于实时绘图的数据缓冲区
        self.plot_data = {
            'x1': deque(maxlen=1000),  # 最多保存1000个点
            'x2': deque(maxlen=1000),
            'cloud_ids': deque(maxlen=1000),
            'rules': []  # 当前规则列表
        }

        # 启动实时绘图线程
        self.start_plotting()

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

        # 更新实时绘图数据
        self.update_plot_data(x_k[0], x_k[1], self.cloud_ids[-1])

        return u_k

    def update_plot_data(self, x1, x2, cloud_id):
        """更新实时绘图数据"""
        with self.plot_lock:
            self.plot_data['x1'].append(x1)
            self.plot_data['x2'].append(x2)
            self.plot_data['cloud_ids'].append(cloud_id)
            # 深拷贝规则列表，避免绘图线程访问时发生变化
            self.plot_data['rules'] = [
                {
                    'mu': rule['mu'].copy(),
                    'theta': rule['theta'].copy()
                } for rule in self.rules
            ]

    def start_plotting(self):
        """启动实时绘图线程"""
        if self.plot_thread is not None and self.plot_thread.is_alive():
            return  # 如果线程已经在运行，则不重复启动

        self.plot_running = True
        self.plot_thread = threading.Thread(target=self.plotting_thread, daemon=True)
        self.plot_thread.start()
        logger.info("实时绘图线程已启动")

    def stop_plotting(self):
        """停止实时绘图线程"""
        self.plot_running = False
        if self.plot_thread is not None:
            self.plot_thread.join(timeout=1.0)
            self.plot_thread = None

    def plotting_thread(self):
        """实时绘图线程函数"""
        # 设置绘图样式
        plt.style.use('ggplot')

        # 创建图形和坐标轴
        fig, ax = plt.subplots(figsize=(10, 8))
        fig.canvas.manager.set_window_title('RECCo控制器 - 云分布实时可视化')

        # 定义颜色映射
        distinct_colors = [
            '#E41A1C',  # 红色
            '#4DAF4A',  # 绿色
            '#377EB8',  # 蓝色
            '#984EA3',  # 紫色
            '#FF7F00',  # 橙色
            '#FFFF33',  # 黄色
            '#A65628',  # 棕色
            '#F781BF',  # 粉色
            '#999999',  # 灰色
            '#66C2A5',  # 青绿色
            '#FC8D62',  # 橙红色
            '#8DA0CB',  # 淡蓝色
            '#E78AC3',  # 淡紫色
            '#A6D854',  # 黄绿色
            '#FFD92F',  # 金黄色
        ]

        # 初始化散点图和云中心
        scatter = ax.scatter([], [], s=10, alpha=0.6)
        centers = ax.scatter([], [], s=200, facecolors='none', edgecolors='none', linewidth=2)

        # 设置坐标轴范围和标签
        ax.set_xlim(-1.0, 1.0)
        ax.set_ylim(-0.2, 1.2)
        ax.set_xlabel(r'$\varepsilon_{k,norm}$', fontsize=12)
        ax.set_ylabel(r'$y^r_{k,norm}$', fontsize=12)
        ax.set_title(r'RECCo控制器 - 云分布实时可视化', fontsize=14)
        ax.grid(True, linestyle=':', alpha=0.3)

        # 添加规则数量文本
        rule_text = ax.text(0.02, 0.98, '规则数量: 0', transform=ax.transAxes,
                            fontsize=12, verticalalignment='top')

        # 添加PID参数文本
        pid_text = ax.text(0.02, 0.93, 'PID参数: P=0.00, I=0.00, D=0.00, R=0.00',
                           transform=ax.transAxes, fontsize=10, verticalalignment='top')

        # 添加血糖信息文本
        bg_text = ax.text(0.02, 0.88, '血糖: 0 mg/dL, 胰岛素: 0.00 U/min',
                          transform=ax.transAxes, fontsize=10, verticalalignment='top')

        # 动画更新函数
        def update(frame):
            with self.plot_lock:
                if len(self.plot_data['x1']) == 0:
                    return scatter, centers, rule_text, pid_text, bg_text

                # 提取数据
                x1_data = list(self.plot_data['x1'])
                x2_data = list(self.plot_data['x2'])
                cloud_ids = list(self.plot_data['cloud_ids'])
                rules = self.plot_data['rules']

                # 准备散点图数据
                x = np.array(x1_data)
                y = np.array(x2_data)
                colors = [distinct_colors[cid % len(distinct_colors)] for cid in cloud_ids]

                # 更新散点图
                scatter.set_offsets(np.column_stack([x, y]))
                scatter.set_color(colors)

                # 更新云中心
                if rules:
                    center_x = [rule['mu'][0] for rule in rules]
                    center_y = [rule['mu'][1] for rule in rules]
                    center_colors = [distinct_colors[i % len(distinct_colors)] for i in range(len(rules))]

                    centers.set_offsets(np.column_stack([center_x, center_y]))
                    centers.set_edgecolors(center_colors)

                # 更新规则数量文本
                rule_text.set_text(f'规则数量: {len(rules)}')

                # 更新PID参数文本 (显示最后一个规则的参数)
                if rules:
                    last_rule = rules[-1]
                    P, I, D, R = last_rule['theta']
                    pid_text.set_text(f'最新规则PID参数: P={P:.4f}, I={I:.4f}, D={D:.4f}, R={R:.4f}')

                # 更新血糖信息文本
                if self.bg_history and self.insulin_history:
                    bg_text.set_text(
                        f'血糖: {self.bg_history[-1]:.1f} mg/dL, 胰岛素: {self.insulin_history[-1]:.4f} U/min')

                # 动态调整坐标轴范围
                if len(x) > 0:
                    x_min, x_max = min(x), max(x)
                    y_min, y_max = min(y), max(y)
                    x_margin = max(0.1, (x_max - x_min) * 0.1)
                    y_margin = max(0.1, (y_max - y_min) * 0.1)
                    ax.set_xlim(x_min - x_margin, x_max + x_margin)
                    ax.set_ylim(y_min - y_margin, y_max + y_margin)

            return scatter, centers, rule_text, pid_text, bg_text

        # 创建动画
        ani = FuncAnimation(fig, update, interval=500, blit=True)

        # 显示图形
        plt.tight_layout()
        plt.show(block=False)

        # 保持线程运行，直到被停止
        while self.plot_running:
            plt.pause(0.5)  # 暂停一小段时间，让matplotlib处理事件

        plt.close(fig)
        logger.info("实时绘图线程已停止")

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
        # 停止当前绘图线程
        self.stop_plotting()

        # 重置控制器状态
        self.init_recco()
        self.bg_history = []
        self.insulin_history = []
        self.reference_history = []
        self.time_history = []
        self.rule_count_history = []

        # 清空绘图数据
        with self.plot_lock:
            self.plot_data['x1'].clear()
            self.plot_data['x2'].clear()
            self.plot_data['cloud_ids'].clear()
            self.plot_data['rules'] = []

        # 重新启动绘图线程
        self.start_plotting()


