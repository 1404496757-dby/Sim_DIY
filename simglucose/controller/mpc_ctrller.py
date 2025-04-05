import numpy as np
from scipy import linalg
from .base import Controller, Action


class MPCController(Controller):
    def __init__(self, patient_model=None, prediction_horizon=5, control_horizon=3,
                 target=110, Q=1, R=0.1, max_insulin=5):
        """
        MPC Controller for glucose regulation

        Parameters:
        - patient_model: Dictionary containing patient-specific model parameters
        - prediction_horizon: Prediction horizon for MPC (in steps)
        - control_horizon: Control horizon for MPC (in steps)
        - target: Target blood glucose level (mg/dL)
        - Q: State cost weight matrix
        - R: Control cost weight matrix
        - max_insulin: Maximum allowed insulin dose (U/min)
        """
        self.prediction_horizon = prediction_horizon
        self.control_horizon = control_horizon
        self.target = target
        self.Q = Q
        self.R = R
        self.max_insulin = max_insulin

        # Default model parameters (can be replaced with patient-specific ones)
        self.model = patient_model or self._get_default_model()

        # Initialize state
        self.x_hat = np.zeros((self.model['A'].shape[0], 1))  # State estimate
        self.last_CGM = None

    def _get_default_model(self):
        """
        Returns default state-space model parameters for UVa/Padova simulator
        This is a simplified linear approximation - real model would need
        to be identified from the simulator
        """
        # Discrete-time state-space model (sample time = 5 min)
        # x[k+1] = A x[k] + B u[k] + Bd d[k]
        # y[k] = C x[k]

        # These matrices would need to be properly identified from the simulator
        # The following are placeholder values for illustration
        A = np.array([[0.9, 0.1, 0],
                      [0, 0.95, 0.05],
                      [0, 0, 0.8]])

        B = np.array([[0.5],
                      [0.3],
                      [0.1]])

        Bd = np.array([[0.2],
                       [0.1],
                       [0.05]])

        C = np.array([[1, 0, 0]])

        return {
            'A': A,
            'B': B,
            'Bd': Bd,
            'C': C,
            'sample_time': 5  # minutes
        }

    def policy(self, observation, reward, done, **info):
        """
        MPC control policy

        Inputs:
        - observation: Namedtuple with CGM measurement (mg/dL)
        - reward: Current reward
        - done: Flag indicating end of episode
        - info: Additional info including patient_name and sample_time

        Returns:
        - action: Controller action (basal, bolus)
        """
        current_CGM = observation.CGM

        # Update state estimate (simplified - would normally use a Kalman filter)
        if self.last_CGM is not None:
            y = np.array([[current_CGM]])
            self.x_hat = self._update_state_estimate(y)
        self.last_CGM = current_CGM

        # Calculate MPC action
        insulin = self._compute_mpc_action(current_CGM)

        # Convert to simglucose action format
        basal = max(0, min(insulin, self.max_insulin))
        bolus = 0  # MPC only handles basal in this simple example

        return Action(basal=basal, bolus=bolus)

    def _update_state_estimate(self, y):
        """
        Simple state estimator (would normally use Kalman filter)
        """
        # Simple observer - in practice would use proper state estimation
        x_pred = self.model['A'] @ self.x_hat
        y_pred = self.model['C'] @ x_pred
        error = y - y_pred
        L = np.array([[0.5], [0.3], [0.2]])  # Observer gain (tuned)

        return x_pred + L * error

    def _compute_mpc_action(self, current_CGM):
        """
        Compute MPC action using quadratic programming
        """
        try:
            from cvxpy import Variable, Minimize, Problem, norm
            cvxpy_available = True
        except ImportError:
            cvxpy_available = False

        if not cvxpy_available:
            # Fallback to simple PID-like control if CVXPY not available
            error = current_CGM - self.target
            if current_CGM > 180:
                return 0.05 * error
            elif current_CGM > 140:
                return 0.02 * error
            elif current_CGM < 70:
                return -0.01 * error
            else:
                return 0

        # Setup MPC optimization problem using CVXPY
        A, B, C = self.model['A'], self.model['B'], self.model['C']
        nx, nu = B.shape  # state and input dimensions

        # Initialize variables
        x = Variable((nx, self.prediction_horizon + 1))
        u = Variable((nu, self.prediction_horizon))

        # Initial state
        constraints = [x[:, 0] == self.x_hat.flatten()]

        # System dynamics constraints
        for k in range(self.prediction_horizon):
            constraints += [x[:, k + 1] == A @ x[:, k] + B @ u[:, k]]

        # Control constraints
        for k in range(self.prediction_horizon):
            constraints += [u[:, k] >= 0, u[:, k] <= self.max_insulin]

        # Cost function
        cost = 0
        for k in range(self.prediction_horizon):
            # Tracking error cost
            cost += self.Q * (C @ x[:, k] - self.target) ** 2
            # Control effort cost
            cost += self.R * (u[:, k]) ** 2

        # Terminal cost
        cost += 10 * self.Q * (C @ x[:, self.prediction_horizon] - self.target) ** 2

        # Solve optimization problem
        prob = Problem(Minimize(cost), constraints)
        prob.solve(solver='ECOS')

        if prob.status != 'optimal':
            return 0  # Fallback to no insulin if optimization fails

        # Return first control action (according to MPC principle)
        return u[:, 0].value[0]

    def reset(self):
        """
        Reset controller state
        """
        self.x_hat = np.zeros((self.model['A'].shape[0], 1))
        self.last_CGM = None