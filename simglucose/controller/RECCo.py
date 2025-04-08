import numpy as np
import pandas as pd
import os
from .base import Controller, Action
import logging

logger = logging.getLogger(__name__)


class CloudManager:
    """数据云管理器，用于存储和管理RECCo控制器的数据云"""

    _instance = None  # 单例模式
    CSV_FILE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cloud_data.csv')

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

        # 尝试从CSV文件加载云数据
        self._load_from_csv()

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

        # 添加新云后保存到CSV
        self._save_to_csv()

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
        # 更新时间后保存到CSV
        self._save_to_csv()

    def get_last_add_time(self):
        """获取上次添加云的时间"""
        return self.last_add_time

    def _save_to_csv(self):
        """将云数据保存到CSV文件"""
        try:
            # 创建一个包含所有云数据的列表
            cloud_data = []

            # 添加管理器状态
            manager_state = {
                'type': 'manager_state',
                'time': self.time,
                'last_add_time': self.last_add_time
            }
            cloud_data.append(manager_state)

            # 添加每个云的数据
            for i, cloud in enumerate(self.clouds):
                cloud_entry = {
                    'type': 'cloud',
                    'index': i,
                    'added_time': cloud['added_time'],
                    'count': cloud['count'],
                    'mean_0': cloud['mean'][0],
                    'mean_1': cloud['mean'][1],
                    'sigma': cloud['sigma'],
                    'param_P': cloud['params'][0],
                    'param_I': cloud['params'][1],
                    'param_D': cloud['params'][2],
                    'param_R': cloud['params'][3]
                }
                cloud_data.append(cloud_entry)

            # 转换为DataFrame并保存
            df = pd.DataFrame(cloud_data)
            df.to_csv(self.CSV_FILE_PATH, index=False)
        except Exception as e:
            print(f"保存云数据到CSV时出错: {e}")

    def _load_from_csv(self):
        """从CSV文件加载云数据"""
        try:
            if not os.path.exists(self.CSV_FILE_PATH):
                print("CSV文件不存在，将创建新的云数据")
                return

            df = pd.read_csv(self.CSV_FILE_PATH)
            if df.empty:
                return

            # 加载管理器状态
            manager_rows = df[df['type'] == 'manager_state']
            if not manager_rows.empty:
                manager_state = manager_rows.iloc[0]
                self.time = manager_state['time']
                self.last_add_time = manager_state['last_add_time']

            # 加载云数据
            cloud_rows = df[df['type'] == 'cloud'].sort_values('index')
            for _, row in cloud_rows.iterrows():
                mean = np.array([row['mean_0'], row['mean_1']])
                params = np.array([row['param_P'], row['param_I'], row['param_D'], row['param_R']])

                cloud = {
                    'mean': mean,
                    'sigma': row['sigma'],
                    'count': row['count'],
                    'params': params,
                    'added_time': row['added_time']
                }
                self.clouds.append(cloud)

            print(f"从CSV加载了 {len(self.clouds)} 个云数据")
        except Exception as e:
            print(f"从CSV加载云数据时出错: {e}")
            # 如果加载失败，使用空的云数据列表
            self.clouds = []

    def update_cloud(self, index, new_params=None, increment_count=True):
        """更新云数据"""
        if 0 <= index < len(self.clouds):
            if increment_count:
                self.clouds[index]['count'] += 1

            if new_params is not None:
                self.clouds[index]['params'] = new_params

            # 更新后保存到CSV
            self._save_to_csv()

    def get_cloud_info(self, detailed=False):
        """
        获取云数据的信息

        参数:
        detailed -- 是否返回详细信息，默认为False

        返回:
        cloud_info -- 包含云数据信息的字典或列表
        """
        if not self.clouds:
            return "没有云数据"

        if not detailed:
            # 返回简要信息
            return {
                "云数量": len(self.clouds),
                "当前时间步": self.time,
                "上次添加时间": self.last_add_time
            }

        # 返回详细信息
        cloud_details = []
        for i, cloud in enumerate(self.clouds):
            cloud_info = {
                "云索引": i,
                "添加时间": cloud['added_time'],
                "样本数": cloud['count'],
                "均值": cloud['mean'].tolist(),
                "参数": {
                    "P": cloud['params'][0],
                    "I": cloud['params'][1],
                    "D": cloud['params'][2],
                    "R": cloud['params'][3]
                }
            }
            cloud_details.append(cloud_info)

        return cloud_details

    def print_cloud_info(self, detailed=False):
        """
        打印云数据信息

        参数:
        detailed -- 是否打印详细信息，默认为False
        """
        info = self.get_cloud_info(detailed)

        if isinstance(info, str):
            print(info)
            return

        if not detailed:
            print(f"云数量: {info['云数量']}")
            print(f"当前时间步: {info['当前时间步']}")
            print(f"上次添加时间: {info['上次添加时间']}")
            return

        print(f"总云数量: {len(info)}")
        for cloud_info in info:
            print("\n" + "=" * 40)
            print(f"云索引: {cloud_info['云索引']}")
            print(f"添加时间: {cloud_info['添加时间']}")
            print(f"样本数: {cloud_info['样本数']}")
            print(f"均值: {cloud_info['均值']}")
            print("PID-R参数:")
            for param, value in cloud_info['参数'].items():
                print(f"  {param}: {value:.6f}")


class RECCoGlucoseController(Controller):
    def __init__(self, target=120, safe_min=140, safe_max=180, Ts=3, tau=40, y_range=(70, 180)):
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
        self.G_sign = 1  # 胰岛素对血糖是负影响
        self.n_add = 20  # 5分钟采样，20个样本约1.5小时

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
            # return np.array([0.0, 0.0, 0.0, 0.0])  # P, I, D, R

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
        E = self.target - CGM

        # 3. 数据预处理
        x = self._normalize(E, y_ref)

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
        elif active_idx >= 0:
            # 更新活跃云的使用计数
            self.cloud_manager.update_cloud(active_idx, increment_count=True)

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
                # delta_P = self.alpha * self.G_sign * abs(E * e) / denom
                # delta_I = self.alpha * self.G_sign * abs(E * Delta_e) / denom
                # delta_D = self.alpha * self.G_sign * abs(E * Delta_e) / denom
                # delta_R = self.alpha * self.G_sign * E / denom

                # 应用泄漏和投影
                new_params = (1 - self.sigma_L) * np.array([P, I, D, R]) + np.array(
                    [delta_P, delta_I, delta_D, delta_R])
                new_params[:3] = np.maximum(new_params[:3], 0)  # P,I,D >=0
                active_cloud['params'] = new_params

                # 更新云参数并保存到CSV
                self.cloud_manager.update_cloud(active_idx, new_params=new_params, increment_count=False)

            # 6. 计算控制量 (分开计算basal和bolus)
            basal_raw = P * e + I * self.Sigma_e + D * Delta_e + R
            # 低血糖安全保护
            if CGM < 140:  # 低血糖保护
                basal_raw = 0
            # elif CGM < 100 and e < 0:  # 接近低血糖且血糖仍在下降
            #     basal_raw = basal_raw * 0.5  # 减少胰岛素剂量
            # 限制在允许范围内
            basal = max(self.u_range[0], min(basal_raw, self.u_range[1]))
            # 大餐时的额外胰岛素 (bolus)
            bolus = 0  # 默认不使用bolus
            # 如果info中包含meal信息，可以考虑添加餐前胰岛素
            if 'meal' in info and info['meal'] > 10:  # 大于10g的碳水被视为餐食
                # 简单的餐前胰岛素计算 (碳水/胰岛素比例约为10:1)
                bolus = info['meal'] / 10 * 0.5  # 每10g碳水给予0.5U胰岛素
                bolus = min(bolus, 5.0)  # 限制最大bolus

        # 更新状态
        self.last_e = e
        # self.last_CGM = CGM
        self.last_action = Action(basal=basal, bolus=bolus)

        if current_time % 20 == 0:
            self.save_cloud_data()

        return self.last_action

    def get_cloud_info(self, detailed=False):
        """
        获取云数据信息

        参数:
        detailed -- 是否返回详细信息，默认为False

        返回:
        cloud_info -- 包含云数据信息的字典或列表
        """
        return self.cloud_manager.get_cloud_info(detailed)

    def print_cloud_info(self, detailed=False):
        """
        打印云数据信息

        参数:
        detailed -- 是否打印详细信息，默认为False
        """
        self.cloud_manager.print_cloud_info(detailed)

    def save_cloud_data(self):
        """手动保存云数据到CSV文件"""
        self.cloud_manager._save_to_csv()

    def reset(self):
        """重置控制器状态，但不重置云数据"""
        # 只重置控制器内部状态，不重置云数据
        self.last_CGM = None
        self.last_action = Action(basal=0, bolus=0)
        self.Sigma_e = 0
        self.last_e = 0
        if hasattr(self, 'y_ref_prev'):
            delattr(self, 'y_ref_prev')

        self.save_cloud_data()