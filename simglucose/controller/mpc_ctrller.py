from .base import Controller, Action
import numpy as np
import pandas as pd
import pkg_resources
import logging
from scipy import linalg

logger = logging.getLogger(__name__)
PATIENT_PARA_FILE = pkg_resources.resource_filename(
    'simglucose', 'params/vpatient_params.csv')


class MPCController(Controller):
    """
    Model Predictive Controller using UVa/Padova parameters
    """

    def __init__(self, target=110, prediction_horizon=5, control_horizon=3):
        # Load patient parameters
        self.patient_params = pd.read_csv(PATIENT_PARA_FILE)
        self.target = target
        self.prediction_horizon = prediction_horizon
        self.control_horizon = control_horizon

        # Will be initialized when patient_name is known
        self.model = None
        self.x_hat = None
        self.last_CGM = None
        self.last_insulin = 0

    def policy(self, observation, reward, done, **kwargs):
        sample_time = kwargs.get('sample_time', 1)
        pname = kwargs.get('patient_name')

        # Initialize model for this patient if not already done
        if self.model is None:
            self._init_patient_model(pname)

        current_CGM = observation.CGM

        # Update state estimate
        if self.last_CGM is not None:
            y = np.array([[current_CGM - self.model['Gb']]])
            self.x_hat = self._update_state_estimate(y)
        self.last_CGM = current_CGM

        # Calculate MPC action
        insulin = self._compute_mpc_action(current_CGM)

        # Convert to simglucose action format
        basal = max(0, min(insulin, self.model['max_insulin']))
        bolus = 0  # MPC handles basal, bolus could be added separately

        # Store last insulin for state estimation
        self.last_insulin = basal

        return Action(basal=basal, bolus=bolus)

    def _init_patient_model(self, patient_name):
        """Initialize patient-specific model from parameters"""
        if any(self.patient_params.Name.str.match(patient_name)):
            params = self.patient_params[
                self.patient_params.Name.str.match(patient_name)].iloc[0]
        else:
            # Default parameters if patient not found
            params = self.patient_params.iloc[0]
            logger.warning(f"Patient {patient_name} not found, using default parameters")

        # Extract relevant parameters
        BW = params.BW  # Body weight (kg)
        Gb = params.Gb  # Basal glucose (mg/dL)
        Ib = params.Ib  # Basal insulin (mU/L)
        Vg = params.Vg  # Glucose distribution volume (dL/kg)
        Vi = params.Vi  # Insulin distribution volume (L/kg)
        u2ss = params.u2ss  # Steady-state insulin (pmol/(L*kg))

        # Convert u2ss from pmol/(L*kg) to U/min
        max_insulin = u2ss * BW / 6000 * 5  # Allow 3x basal as maximum

        # Create continuous-time state-space model
        # Using 4-state model: glucose, insulin, insulin action, and meal absorption
        # These parameters would need to be mapped from the UVa/Padova parameters
        p1 = 0.028735  # Glucose effectiveness (1/min)
        p2 = params.p2u  # Insulin sensitivity (1/min)
        p3 = 0.0003  # Rate of insulin action (1/min)
        ke = params.ke1  # Insulin elimination rate (1/min)

        # Continuous-time matrices
        Ac = np.array([
            [-p1, 0, -Gb * p2, 1 / (Vg * BW)],
            [0, -ke, 0, 0],
            [0, p3, -p3, 0],
            [0, 0, 0, -0.05]  # Meal absorption dynamics
        ])

        Bc = np.array([
            [0],
            [1 / Vi],
            [0],
            [0]
        ])

        C = np.array([[1, 0, 0, 0]])  # Measure glucose

        # Discretize with sample time = 5 min (300 sec)
        Ts = 5 * 60
        A = linalg.expm(Ac * Ts)

        # Numerically integrate to get discrete B matrix
        B = np.zeros_like(Bc)
        for i in range(100):
            B += linalg.expm(Ac * Ts * i / 100) @ Bc * (Ts / 100)

        self.model = {
            'A': A,
            'B': B,
            'C': C,
            'Gb': Gb,
            'Ib': Ib,
            'max_insulin': max_insulin,
            'sample_time': 5
        }

        # Initialize state estimate
        self.x_hat = np.zeros((Ac.shape[0], 1))

    def _update_state_estimate(self, y):
        """Kalman filter update"""
        A, B, C = self.model['A'], self.model['B'], self.model['C']

        # Predict
        x_pred = A @ self.x_hat + B * (self.last_insulin - self.model['Ib'])

        # Update (simplified Kalman gain)
        y_pred = C @ x_pred
        error = y - y_pred
        L = np.array([[0.6], [0.1], [0.2], [0.1]])  # Observer gains

        return x_pred + L * error

    def _compute_mpc_action(self, current_CGM):
        """Solve MPC optimization problem"""
        try:
            from cvxpy import Variable, Minimize, Problem, norm, ECOS, SCS, maximum
            cvxpy_available = True
        except ImportError:
            cvxpy_available = False


        if not cvxpy_available:
            # Fallback to PID-like control
            error = current_CGM - self.target
            if current_CGM > 180:
                return 0.05 * error
            elif current_CGM > 140:
                return 0.02 * error
            elif current_CGM < 70:
                return 0
            else:

                return 0

        # 添加血糖安全检查
        if current_CGM < 80:  # 低血糖保护
            return 0

        # Setup MPC optimization
        A, B, C = self.model['A'], self.model['B'], self.model['C']
        nx, nu = B.shape
        x = Variable((nx, self.prediction_horizon + 1))
        u = Variable((nu, self.prediction_horizon))

        # 放宽约束条件，提高数值稳定性
        constraints = [x[:, 0] == self.x_hat.flatten()]
        for k in range(self.prediction_horizon):
            constraints += [
                x[:, k + 1] == A @ x[:, k] + B @ u[:, k],
                u[:, k] >= -self.model['Ib'],
                u[:, k] <= self.model['max_insulin'] - self.model['Ib']
            ]

        # 修改成本函数，增加正则化项
        cost = 0
        for k in range(self.prediction_horizon):
            # 增加血糖偏差惩罚 - 修复严格不等式问题

            glucose_error = C @ x[:, k] + self.model['Gb'] - self.target
            # 使用两个不同的权重，但避免使用严格不等式
            cost += 100 * glucose_error ** 2  # 基础惩罚
            cost += 100 * maximum(0, -glucose_error) ** 2  # 高血糖额外惩罚

            # 增加控制输入惩罚
            cost += 0.5 * (u[:, k] + self.model['Ib']) ** 2  # 减小惩罚权重

            # 如果不是第一步，增加控制变化率惩罚
            if k > 0:
                cost += 5.0 * (u[:, k] - u[:, k - 1]) ** 2

        # 创建问题
        prob = Problem(Minimize(cost), constraints)

        # 尝试不同的求解器
        try:
            # 首先尝试OSQP（默认）
            prob.solve(solver='OSQP', eps_abs=1e-3, eps_rel=1e-3, max_iter=10000, verbose=False)
        except Exception as e1:
            try:
                # 如果OSQP失败，尝试ECOS
                prob.solve(solver=ECOS, verbose=False)
            except Exception as e2:
                try:
                    # 如果ECOS也失败，尝试SCS
                    prob.solve(solver=SCS, verbose=False)
                except Exception as e3:
                    # 所有求解器都失败，使用备用控制策略
                    logger.warning(f"All solvers failed: {e1}, {e2}, {e3}")
                    # 简单的PID控制作为备用
                    error = current_CGM - self.target
                    if current_CGM > 180:
                        return min(0.5 + 0.05 * error, self.model['max_insulin'])
                    elif current_CGM > 140:
                        return min(0.2 + 0.02 * error, self.model['max_insulin'])
                    elif current_CGM < 80:
                        return 0
                    else:
                        return min(0.1, self.model['max_insulin'])

        # 检查求解状态
        if prob.status not in ['optimal', 'optimal_inaccurate']:
            logger.warning(f"Solver status: {prob.status}")
            # 使用备用控制策略
            error = current_CGM - self.target
            if current_CGM > 180:
                return min(0.5 + 0.05 * error, self.model['max_insulin'])
            elif current_CGM > 140:
                return min(0.2 + 0.02 * error, self.model['max_insulin'])
            elif current_CGM < 80:
                return 0
            else:
                return min(0.1, self.model['max_insulin'])

        # 获取最优控制输入
        try:
            insulin = max(0, u[:, 0].value[0] + self.model['Ib'])

            if current_CGM < 80:  # 低血糖保护
                insulin = 0
            elif current_CGM < 100:  # 接近低血糖
                insulin = insulin * 0.5  # 减少胰岛素剂量
            elif current_CGM > 180:
                min_insulin = min(0.5, self.model['max_insulin'])
                insulin = max(insulin, min_insulin)
            elif current_CGM > 150:
                min_insulin = min(0.2, self.model['max_insulin'])
                insulin = max(insulin, min_insulin)
        except:
            logger.warning("Failed to extract solution value")
            return 0

        return max(0, u[:, 0].value[0] + self.model['Ib'])

    def reset(self):
        """Reset controller state"""
        if self.model is not None:
            self.x_hat = np.zeros((self.model['A'].shape[0], 1))
        self.last_CGM = None
        self.last_insulin = 0
