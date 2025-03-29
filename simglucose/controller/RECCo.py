from .base import Controller, Action
import numpy as np
from collections import defaultdict


class RECCoController(Controller):
    def __init__(self, target=140, a_r=0.9, gamma_p=0.1, d_dead=10):
        self.target = target
        self.a_r = a_r
        self.gamma_p = gamma_p
        self.d_dead = d_dead

        # RECCo-specific states
        self.clouds = []  # List of data clouds
        self.mu_G = None  # Global mean
        self.Sigma_G = 0  # Global scalar product
        self.k = 0  # Time step counter
        self.y_r_prev = target  # Previous reference model output

        # PID states
        self.integrated_error = 0
        self.prev_error = 0

    def policy(self, observation, reward, done, **info):
        # Extract inputs
        bg = observation.CGM
        meal = info.get('meal', 0)
        sample_time = info.get('sample_time', 1)

        # 1. Update reference model
        y_r = self.a_r * self.y_r_prev + (1 - self.a_r) * self.target
        self.y_r_prev = y_r

        # 2. Calculate tracking error
        error = bg - y_r

        # 3. Initialize 'u' with a default value
        u = 0.0  # Default control signal

        # 4. RECCo core logic
        if self.k == 0:
            self._add_cloud(bg, meal)
        else:
            u = self._recco_control(bg, error, sample_time)
            self._update_structure(bg, u, sample_time)

        # 5. Generate insulin action
        basal = max(u, 0)  # Ensure non-negative basal
        bolus = 0 if meal == 0 else basal * 0.5  # Simple bolus heuristic
        return Action(basal=basal, bolus=bolus)

    def _recco_control(self, bg, error, sample_time):
        """Calculate RECCo control signal."""
        if not self.clouds:
            return 0.0  # Fallback if no clouds exist

        densities = [self._calculate_density(bg, cloud) for cloud in self.clouds]
        sum_densities = sum(densities)
        weights = [d / max(sum_densities, 1e-6) for d in densities]

        u = 0
        for i, cloud in enumerate(self.clouds):
            P_term = cloud['P'] * error
            self.integrated_error += error * sample_time
            D_term = (error - self.prev_error) / sample_time

            if abs(error) > self.d_dead:
                cloud['P'] += self.gamma_p * weights[i] * error

            u += weights[i] * (P_term + cloud['I'] * self.integrated_error + cloud['D'] * D_term)

        self.prev_error = error
        return u

    def _calculate_density(self, x, cloud):
        return 1 / (1 + np.linalg.norm(x - cloud['mu']) ** 2)

    def _update_structure(self, x, u, sample_time):
        z = np.array([x, u])
        gamma_k = self._calculate_global_density(z)

        if gamma_k > max([self._calculate_global_density(np.array([c['mu'], 0])) for c in self.clouds], default=0):
            self._add_cloud(x, u)

        self.k += 1
        if self.k == 1:
            self.mu_G = z
            self.Sigma_G = np.linalg.norm(z) ** 2
        else:
            self.mu_G = (self.k - 1) / self.k * self.mu_G + z / self.k
            self.Sigma_G = (self.k - 1) / self.k * self.Sigma_G + np.linalg.norm(z) ** 2 / self.k

    def _calculate_global_density(self, z):
        if self.k == 0:
            return 1.0
        return 1 / (1 + np.linalg.norm(z - self.mu_G) ** 2 + self.Sigma_G - np.linalg.norm(self.mu_G) ** 2)

    def _add_cloud(self, x, u):
        new_cloud = {
            'mu': x,
            'Sigma': 1.0,
            'P': 0.1,
            'I': 0.001,
            'D': 0.01,
            'radius': 1.0
        }
        self.clouds.append(new_cloud)

    def reset(self):
        self.clouds = []
        self.mu_G = None
        self.Sigma_G = 0
        self.k = 0
        self.y_r_prev = self.target
        self.integrated_error = 0
        self.prev_error = 0