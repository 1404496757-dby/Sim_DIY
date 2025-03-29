from .base import Controller, Action
import numpy as np
from collections import defaultdict


class RECCoController(Controller):
    """
    RECCo (Robust Evolving Cloud-based Controller) for glucose control.
    Combines data cloud-based fuzzy rules with adaptive PID control.
    """

    def __init__(self, target=140, a_r=0.9, gamma_p=0.1, d_dead=10):
        """
        Args:
            target (float): Target glucose level (mg/dL).
            a_r (float): Reference model pole (0 < a_r < 1).
            gamma_p (float): Adaptive gain for parameter adjustment.
            d_dead (float): Dead zone threshold for tracking error.
        """
        self.target = target
        self.a_r = a_r
        self.gamma_p = gamma_p
        self.d_dead = d_dead

        # RECCo-specific states
        self.clouds = []  # List of data clouds: {'mu', 'Sigma', 'P', 'radius'}
        self.mu_G = None  # Global mean
        self.Sigma_G = 0  # Global scalar product
        self.k = 0  # Time step counter
        self.y_r_prev = target  # Previous reference model output

        # PID states
        self.integrated_error = 0
        self.prev_error = 0

    def policy(self, observation, reward, done, **info):
        """
        RECCo control policy.
        Inputs:
            observation (namedtuple): Contains CGM (mg/dL) and CHO (g/min).
            reward (float): Current reward signal.
            done (bool): Episode termination flag.
            info (dict): Additional info (e.g., meal, sample_time).
        Output:
            Action(basal, bolus): Insulin rates (U/min).
        """
        # Extract inputs
        bg = observation.CGM
        meal = info.get('meal', 0)  # Meal disturbance (g/min)
        sample_time = info.get('sample_time', 1)  # Time step (min)

        # 1. Update reference model (smooth target)
        y_r = self.a_r * self.y_r_prev + (1 - self.a_r) * self.target
        self.y_r_prev = y_r

        # 2. Calculate tracking error
        error = bg - y_r

        # 3. RECCo core logic
        if self.k == 0:
            # Initialize first cloud
            self._add_cloud(bg, meal)
        else:
            # Calculate control action
            u = self._recco_control(bg, error, sample_time)

            # Structure evolution
            self._update_structure(bg, u, sample_time)

        # 4. Generate insulin action
        basal = max(u, 0)  # Ensure non-negative basal
        bolus = 0 if meal == 0 else basal * 0.5  # Simple bolus heuristic
        return Action(basal=basal, bolus=bolus)

    def _recco_control(self, bg, error, sample_time):
        """
        RECCo control signal calculation.
        """
        # Calculate local densities (simplified)
        densities = [self._calculate_density(bg, cloud) for cloud in self.clouds]
        sum_densities = sum(densities)
        weights = [d / max(sum_densities, 1e-6) for d in densities]

        # PID-like control per cloud
        u = 0
        for i, cloud in enumerate(self.clouds):
            # Update PID terms
            P_term = cloud['P'] * error
            self.integrated_error += error * sample_time
            D_term = (error - self.prev_error) / sample_time

            # Adaptive parameter adjustment (simplified)
            if abs(error) > self.d_dead:
                cloud['P'] += self.gamma_p * weights[i] * error

            # Weighted control signal
            u += weights[i] * (P_term + cloud['I'] * self.integrated_error + cloud['D'] * D_term)

        self.prev_error = error
        return u

    def _calculate_density(self, x, cloud):
        """Calculate local density for a data cloud."""
        return 1 / (1 + np.linalg.norm(x - cloud['mu']) ** 2)

    def _update_structure(self, x, u, sample_time):
        """Update data cloud structure based on new data."""
        z = np.array([x, u])
        gamma_k = self._calculate_global_density(z)

        # Add new cloud if density is significantly different
        if gamma_k > max([self._calculate_global_density(np.array([c['mu'], 0])) for c in self.clouds], default=0):
            self._add_cloud(x, u)

        # Update global statistics
        self.k += 1
        if self.k == 1:
            self.mu_G = z
            self.Sigma_G = np.linalg.norm(z) ** 2
        else:
            self.mu_G = (self.k - 1) / self.k * self.mu_G + z / self.k
            self.Sigma_G = (self.k - 1) / self.k * self.Sigma_G + np.linalg.norm(z) ** 2 / self.k

    def _calculate_global_density(self, z):
        """Calculate global density for structure evolution."""
        if self.k == 0:
            return 1.0
        return 1 / (1 + np.linalg.norm(z - self.mu_G) ** 2 + self.Sigma_G - np.linalg.norm(self.mu_G) ** 2)

    def _add_cloud(self, x, u):
        """Initialize a new data cloud."""
        new_cloud = {
            'mu': x,
            'Sigma': 1.0,
            'P': 0.1,  # Initial proportional gain
            'I': 0.001,  # Initial integral gain
            'D': 0.01,  # Initial derivative gain
            'radius': 1.0
        }
        self.clouds.append(new_cloud)

    def reset(self):
        """Reset controller states."""
        self.clouds = []
        self.mu_G = None
        self.Sigma_G = 0
        self.k = 0
        self.y_r_prev = self.target
        self.integrated_error = 0
        self.prev_error = 0