from .base import Controller
from .base import Action
import numpy as np
from scipy.optimize import minimize
import logging

logger = logging.getLogger(__name__)


class MPCController(Controller):
    def __init__(self, patient_params, prediction_horizon=5, control_horizon=3,
                 target=140, Q=1, R=0.1, max_insulin=5):
        """
        MPC Controller for glucose regulation

        Parameters:
        - patient_params: Dictionary of patient parameters from the CSV
        - prediction_horizon: Number of steps to predict ahead
        - control_horizon: Number of control steps to optimize
        - target: Target blood glucose level (mg/dL)
        - Q: Weight for glucose tracking error
        - R: Weight for control action (insulin)
        - max_insulin: Maximum allowed insulin dose (U/min)
        """
        self.patient_params = patient_params
        self.prediction_horizon = prediction_horizon
        self.control_horizon = control_horizon
        self.target = target
        self.Q = Q
        self.R = R
        self.max_insulin = max_insulin

        # State history for the observer
        self.state_history = []
        self.insulin_history = []
        self.meal_history = []

        # Initialize patient model parameters
        self._init_patient_model()

    def _init_patient_model(self):
        """Extract relevant parameters from patient data for the MPC model"""
        # These are simplified model parameters based on the CSV columns
        self.SI = self.patient_params['ki']  # Insulin sensitivity (1/(mU·min))
        self.ke = self.patient_params['ke1']  # Insulin elimination rate (1/min)
        self.kabs = self.patient_params['kabs']  # Carbohydrate absorption rate (1/min)
        self.kg = self.patient_params['ksc']  # Glucose disappearance rate (1/min)
        self.Vg = self.patient_params['Vg']  # Glucose distribution volume (dL)

        # State vector: [Glucose, Insulin, Carbs]
        self.x = np.zeros(3)

    def _patient_model(self, x, u, d, dt):
        """
        Simplified glucose-insulin-carbohydrate model for prediction
        x: state vector [glucose, insulin, carbs]
        u: insulin input (basal rate)
        d: meal disturbance (carbs)
        dt: time step (min)
        """
        glucose, insulin, carbs = x

        # Glucose dynamics
        dgdt = -self.kg * glucose - self.SI * insulin * glucose + self.kabs * carbs / self.Vg

        # Insulin dynamics
        didt = -self.ke * insulin + u

        # Carbohydrate dynamics
        dcdt = -self.kabs * carbs + d

        # Update state
        new_glucose = glucose + dgdt * dt
        new_insulin = insulin + didt * dt
        new_carbs = carbs + dcdt * dt

        return np.array([new_glucose, new_insulin, new_carbs])

    def _predict(self, x0, u_sequence, d_sequence, dt):
        """
        Predict future states given control sequence
        """
        predictions = []
        x = x0.copy()

        for i in range(self.prediction_horizon):
            if i < len(u_sequence):
                u = u_sequence[i]
                d = d_sequence[i] if i < len(d_sequence) else 0
            else:
                u = u_sequence[-1]  # Use last control input
                d = 0  # Assume no meals beyond known horizon

            x = self._patient_model(x, u, d, dt)
            predictions.append(x[0])  # Only track glucose prediction

        return np.array(predictions)

    def _cost_function(self, u_sequence, x0, d_sequence, dt):
        """
        Cost function for MPC optimization
        """
        # Pad control sequence with last value if shorter than prediction horizon
        if len(u_sequence) < self.prediction_horizon:
            u_sequence = np.concatenate([
                u_sequence,
                np.ones(self.prediction_horizon - len(u_sequence)) * u_sequence[-1]
            ])

        # Get predictions
        predictions = self._predict(x0, u_sequence, d_sequence, dt)

        # Calculate tracking error cost
        error = predictions - self.target
        tracking_cost = self.Q * np.sum(error ** 2)

        # Calculate control effort cost
        control_cost = self.R * np.sum(np.array(u_sequence) ** 2)

        return tracking_cost + control_cost

    def policy(self, observation, reward, done, **kwargs):
        """
        MPC control policy
        """
        # Get current state and parameters
        current_glucose = observation.CGM
        sample_time = kwargs.get('sample_time', 5)  # Default to 5 min if not provided
        meal = kwargs.get('meal', 0)  # Current meal disturbance

        # Update state history
        self.state_history.append(current_glucose)
        self.meal_history.append(meal)

        # Estimate current insulin (simplified - in practice would need better observer)
        if len(self.insulin_history) > 0:
            current_insulin = self.insulin_history[-1]
        else:
            current_insulin = 0

        # Current state vector [glucose, insulin, carbs]
        x0 = np.array([current_glucose, current_insulin, 0])  # Carbs starts at 0

        # Known meal disturbances (simplified - assumes we know upcoming meals)
        d_sequence = self.meal_history[-self.control_horizon:] + [0] * (self.control_horizon - len(self.meal_history))

        # Initial guess for control sequence (previous insulin or basal rate)
        if len(self.insulin_history) > 0:
            u_init = np.ones(self.control_horizon) * self.insulin_history[-1]
        else:
            u_init = np.ones(self.control_horizon) * self.patient_params['Ib'] / 1440  # Convert daily basal to per-min

        # Constraints - insulin must be positive and below max
        bounds = [(0, self.max_insulin)] * self.control_horizon

        # Optimize control sequence
        res = minimize(
            self._cost_function,
            u_init,
            args=(x0, d_sequence, sample_time),
            bounds=bounds,
            method='SLSQP'
        )

        if not res.success:
            logger.warning(f"MPC optimization failed: {res.message}")
            optimal_u = u_init  # Fall back to initial guess
        else:
            optimal_u = res.x

        # Apply first control input
        control_input = optimal_u[0]

        # Update insulin history
        self.insulin_history.append(control_input)

        logger.info(f"MPC control: glucose={current_glucose}, insulin={control_input:.2f}")

        # Return action (convert to basal rate, bolus=0 for MPC)
        action = Action(basal=control_input, bolus=0)
        return action

    def reset(self):
        """Reset controller state"""
        self.state_history = []
        self.insulin_history = []
        self.meal_history = []
        self._init_patient_model()
