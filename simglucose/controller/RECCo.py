import numpy as np
from .base import Controller
from .base import Action
import logging

logger = logging.getLogger(__name__)


class RECCoController(Controller):
    def __init__(self, target=140, u_min=0, u_max=20, y_min=0, y_max=300,
                 tau=40, sample_time=3, G_sign=1):
        """
        Initialize RECCo controller

        Args:
            target: desired blood glucose level (mg/dL)
            u_min: minimum insulin input
            u_max: maximum insulin input
            y_min: minimum possible glucose reading
            y_max: maximum possible glucose reading
            tau: estimated time constant of the system (minutes)
            sample_time: time between measurements (minutes)
            G_sign: known sign of the process gain (+1 or -1)
        """
        self.target = target
        self.u_min = u_min
        self.u_max = u_max
        self.y_min = y_min
        self.y_max = y_max
        self.tau = tau
        self.sample_time = sample_time
        self.G_sign = G_sign

        # Reference model parameters
        self.a_r = 1 - (sample_time / tau)

        # Evolving law parameters
        self.gamma_max = 0.93  # Fixed threshold for adding new clouds
        self.n_add = 20  # Minimum time between adding clouds
        self.last_add_time = -np.inf

        # Adaptation law parameters
        self.alpha_P = 0.1 * (u_max - u_min) / 20
        self.alpha_I = 0.1 * (u_max - u_min) / 20
        self.alpha_D = 0.1 * (u_max - u_min) / 20
        self.alpha_R = 0.1 * (u_max - u_min) / 20

        # Robustness parameters
        self.d_dead = 5  # Dead zone threshold (mg/dL)
        self.sigma_L = 1e-6  # Leakage factor

        # Initialize clouds (empty at start)
        self.clouds = []

        # Tracking variables
        self.y_k_prev = None  # Previous plant output
        self.y_r_prev = None  # Previous reference model output
        self.e_k_prev = None  # Previous tracking error
        self.Sigma_e = 0  # Integral of tracking error
        self.r_k_prev = None  # Previous reference signal

    def policy(self, observation, reward, done, **kwargs):
        # Get current blood glucose reading
        y_k = observation.CGM
        r_k = self.target

        # Initialize on first call
        if self.y_k_prev is None:
            self.y_k_prev = y_k
            self.y_r_prev = y_k
            self.e_k_prev = 0
            self.r_k_prev = r_k
            return Action(basal=0, bolus=0)

        # 1. Reference model update
        y_r_k = self.a_r * self.y_r_prev + (1 - self.a_r) * r_k

        # 2. Calculate tracking error
        e_k = y_r_k - y_k
        Delta_e = e_k - self.e_k_prev



        # 3. Create normalized data vector (2D)
        Delta_y = self.y_max - self.y_min
        Delta_e_norm = Delta_y / 2
        x_k = np.array([
            e_k / Delta_e_norm,
            (y_r_k - self.y_min) / Delta_y
        ])

        # 4. Evolving law - update or add clouds
        active_cloud_idx = self._update_clouds(x_k)

        # 5. Adaptation law - update PID-R parameters of active cloud
        if active_cloud_idx is not None:
            self._adapt_parameters(active_cloud_idx, e_k, Delta_e, r_k)

        # 6. Calculate control signal
        u_k = self._calculate_control_signal(e_k, Delta_e)

        # 7. Apply output constraints
        u_k = np.clip(u_k, self.u_min, self.u_max)

        # Update previous values
        self.y_k_prev = y_k
        self.y_r_prev = y_r_k
        self.e_k_prev = e_k
        self.r_k_prev = r_k
        self.u_k_prev = u_k

        # Update integral term with anti-windup
        if self.u_min < self.u_k_prev < self.u_max:
            self.Sigma_e += e_k * self.sample_time

        logger.info(f'Control input: {u_k}')
        return Action(basal=u_k, bolus=0)

    def _update_clouds(self, x_k):
        """Update cloud structure and return index of active cloud"""
        if not self.clouds:
            # First data point - create initial cloud
            self._add_cloud(x_k)
            return 0

        # Calculate local densities for all clouds
        gamma = []
        for cloud in self.clouds:
            mu_i = cloud['mu']
            sigma_i = cloud['sigma']
            M_i = cloud['M']

            # Calculate local density (6)
            gamma_i = 1 / (1 + np.linalg.norm(x_k - mu_i) ** 2 + sigma_i - np.linalg.norm(mu_i) ** 2)
            gamma.append(gamma_i)

        max_gamma = max(gamma)
        active_cloud_idx = gamma.index(max_gamma)

        # Check if we should add a new cloud
        current_time = len(self.clouds)  # Simplified - should use actual time
        if (max_gamma < self.gamma_max and
                current_time - self.last_add_time >= self.n_add):
            self._add_cloud(x_k)
            active_cloud_idx = len(self.clouds) - 1
            self.last_add_time = current_time
        else:
            # Update active cloud statistics
            cloud = self.clouds[active_cloud_idx]
            M_i = cloud['M']

            # Update mean (7)
            cloud['mu'] = (M_i - 1) / M_i * cloud['mu'] + 1 / M_i * x_k

            # Update mean-square length (8)
            cloud['sigma'] = (M_i - 1) / M_i * cloud['sigma'] + 1 / M_i * np.linalg.norm(x_k) ** 2

            cloud['M'] += 1

        return active_cloud_idx

    def _add_cloud(self, x_k):
        """Add a new cloud to the structure"""
        # Initialize cloud properties
        new_cloud = {
            'mu': x_k,  # Mean value
            'sigma': np.linalg.norm(x_k) ** 2,  # Mean-square length
            'M': 1,  # Number of data points
            'k_add': len(self.clouds),  # Time stamp when added
            'theta': np.zeros(4)  # PID-R parameters [P, I, D, R]
        }

        # If not first cloud, initialize parameters as weighted mean of existing clouds
        if self.clouds:
            lambdas = [gamma / sum(gamma) for gamma in self._calculate_lambdas(x_k)]
            new_cloud['theta'] = sum(l * cloud['theta']
                                     for l, cloud in zip(lambdas, self.clouds))

        self.clouds.append(new_cloud)

    def _calculate_lambdas(self, x_k):
        """Calculate normalized relative densities (5)"""
        gamma = []
        for cloud in self.clouds:
            mu_i = cloud['mu']
            sigma_i = cloud['sigma']
            gamma_i = 1 / (1 + np.linalg.norm(x_k - mu_i) ** 2 + sigma_i - np.linalg.norm(mu_i) ** 2)
            gamma.append(gamma_i)
        return gamma

    def _adapt_parameters(self, cloud_idx, e_k, Delta_e, r_k):
        """Adapt PID-R parameters of the specified cloud (14)"""
        cloud = self.clouds[cloud_idx]
        theta = cloud['theta']
        lambda_k = self._calculate_lambdas(self._get_current_x())[cloud_idx]

        # Dead zone check (17)
        if abs(e_k) < self.d_dead:
            return

        # Calculate parameter updates
        denom = 1 + r_k ** 2

        # For first 5*tau samples, use absolute values
        if len(self.clouds) < 5 * self.tau / self.sample_time:
            delta_P = self.alpha_P * self.G_sign * lambda_k * abs(e_k * e_k) / denom
            delta_I = self.alpha_I * self.G_sign * lambda_k * abs(e_k * Delta_e) / denom
            delta_D = self.alpha_D * self.G_sign * lambda_k * abs(e_k * Delta_e) / denom
            delta_R = self.alpha_R * self.G_sign * lambda_k * e_k / denom
        else:
            delta_P = self.alpha_P * self.G_sign * lambda_k * e_k * e_k / denom
            delta_I = self.alpha_I * self.G_sign * lambda_k * e_k * Delta_e / denom
            delta_D = self.alpha_D * self.G_sign * lambda_k * e_k * Delta_e / denom
            delta_R = self.alpha_R * self.G_sign * lambda_k * e_k / denom

        # Apply leakage (19)
        theta = (1 - self.sigma_L) * theta

        # Update parameters with projection (18)
        # For P, I, D: lower bound = 0, upper bound = infinity
        # For R: no bounds
        theta[0] = max(0, theta[0] + delta_P)  # P
        theta[1] = max(0, theta[1] + delta_I)  # I
        theta[2] = max(0, theta[2] + delta_D)  # D
        theta[3] = theta[3] + delta_R  # R

        cloud['theta'] = theta

    def _calculate_control_signal(self, e_k, Delta_e):
        """Calculate control signal using weighted average of cloud contributions (16)"""
        if not self.clouds:
            return 0

        x_k = self._get_current_x()
        lambdas = self._calculate_lambdas(x_k)
        sum_lambda = sum(lambdas)

        if sum_lambda == 0:
            return 0

        u_total = 0
        for i, cloud in enumerate(self.clouds):
            theta = cloud['theta']
            P, I, D, R = theta
            u_i = P * e_k + I * self.Sigma_e + D * Delta_e + R
            u_total += (lambdas[i] / sum_lambda) * u_i

        return u_total

    def _get_current_x(self):
        """Get current normalized data vector"""
        Delta_y = self.y_max - self.y_min
        Delta_e_norm = Delta_y / 2
        return np.array([
            self.e_k_prev / Delta_e_norm,
            (self.y_r_prev - self.y_min) / Delta_y
        ])

    def reset(self):
        """Reset controller state"""
        self.clouds = []
        self.y_k_prev = None
        self.y_r_prev = None
        self.e_k_prev = None
        self.Sigma_e = 0
        self.r_k_prev = None
        self.u_k_prev = None