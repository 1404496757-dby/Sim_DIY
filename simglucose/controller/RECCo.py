from .base import Controller
from .base import Action
import numpy as np
import logging

logger = logging.getLogger(__name__)


class RECCoController(Controller):
    """
    RECCo Controller implementation based on the Sorensen diabetic model
    with ANYA fuzzy rule-based system and online adaptation.
    """

    def __init__(self, target=100, sample_time=5):
        """
        Initialize RECCo controller parameters
        :param target: Target blood glucose level (mg/dl)
        :param sample_time: Control update frequency (minutes)
        """
        super().__init__(init_state={
            'clouds': [],
            'adapt_params': {
                'P': [], 'I': [], 'D': [], 'R': [],
                'gamma': 0.1, 'lambda': 0.5, 'sigma': 0.1
            },
            'tracking_error': 0.0,
            'prev_error': 0.0,
            'integral_sum': 0.0
        })

        # Controller parameters
        self.target = target
        self.sample_time = sample_time
        self.umax = 100  # Maximum insulin rate (U/min)
        self.umin = 0  # Minimum insulin rate (U/min)
        self.y_min = 20  # Minimum glucose (mg/dl)
        self.y_max = 200  # Maximum glucose (mg/dl)
        self.dead_zone = 0.1
        self.gamma_max = 0.8
        self.n_add = 20

        # Initialize reference model
        self.a_r = 0.93  # Pole parameter
        self.tau = 40  # Time constant

    def policy(self, observation, reward, done, **kwargs):
        """
        Main control policy implementation
        :param observation: Current glucose reading
        :param reward: Current reward value
        :param done: Episode termination flag
        :param kwargs: Additional parameters (patient_name, sample_time)
        :return: Action tuple (basal, bolus)
        """
        # Extract necessary parameters
        patient_name = kwargs.get('patient_name', 'default')
        meal = kwargs.get('meal', 0.0)  # g/min
        env_sample_time = kwargs.get('sample_time', self.sample_time)

        # Get current glucose value
        current_glucose = observation.CGM

        # Calculate control action
        control_signal = self._recco_control(current_glucose, meal, env_sample_time)

        # Apply saturation
        control_signal = np.clip(control_signal, self.umin, self.umax)

        # Split into basal and bolus (assuming bolus is meal-related)
        basal = control_signal
        bolus = 0.0  # Meal bolus handled separately in this example

        return Action(basal=basal, bolus=bolus)

    def _recco_control(self, glucose, meal, sample_time):
        """
        Core RECCo control algorithm
        :param glucose: Current blood glucose (mg/dl)
        :param meal: Current meal intake (g/min)
        :param sample_time: Control interval (minutes)
        :return: Calculated insulin rate (U/min)
        """
        # Update reference model
        ref_model_output = self._update_reference_model(glucose)

        # Calculate tracking error
        error = ref_model_output - glucose
        delta_error = error - self.state['tracking_error']
        self.state['tracking_error'] = error

        # Update integral sum with anti-windup
        if self.umin < self.state['control_signal'] < self.umax:
            self.state['integral_sum'] += error * sample_time
        else:
            self.state['integral_sum'] = np.clip(self.state['integral_sum'] + error * sample_time,
                                                 -self.umax, self.umax)

        # Normalize input space
        x = self._normalize_input(error, delta_error, glucose)

        # Update clouds and fuzzy rules
        self._update_clouds(x)

        # Calculate control signal using fuzzy rules
        control_signal = self._fuzzy_inference(x)

        # Apply protection mechanisms
        control_signal = self._apply_protection(control_signal)

        return control_signal

    def _update_reference_model(self, glucose):
        """
        First-order reference model update
        :param glucose: Current glucose value
        :return: Updated reference output
        """
        self.state['ref_model_output'] = self.a_r * self.state.get('ref_model_output', self.target) + \
                                         (1 - self.a_r) * self.target
        return self.state['ref_model_output']

    def _normalize_input(self, error, delta_error, glucose):
        """
        Normalize input variables for fuzzy system
        :param error: Tracking error
        :param delta_error: Error derivative
        :param glucose: Current glucose
        :return: Normalized input vector
        """
        norm_error = (error) / (self.y_max - self.y_min)
        norm_delta_error = (delta_error) / (self.y_max - self.y_min)
        norm_glucose = (glucose - self.y_min) / (self.y_max - self.y_min)
        return np.array([norm_error, norm_delta_error, norm_glucose])

    def _update_clouds(self, x):
        """
        Update cloud points and fuzzy rules
        :param x: Normalized input vector
        """
        # Calculate distances to existing clouds
        distances = [np.linalg.norm(x - cloud['mu']) for cloud in self.state['clouds']]

        # Check if new cloud is needed
        if not self.state['clouds'] or min(distances) > self.gamma_max:
            # Create new cloud
            new_cloud = {
                'mu': x,
                'sigma': np.linalg.norm(x) ** 2,
                'local_density': 1.0,
                'parameters': {
                    'P': 0.1,
                    'I': 0.05,
                    'D': 0.01,
                    'R': 0.0
                }
            }
            self.state['clouds'].append(new_cloud)
        else:
            # Update closest cloud
            closest_idx = np.argmin(distances)
            self.state['clouds'][closest_idx]['mu'] = (self.state['clouds'][closest_idx]['mu'] *
                                                       (self.state['clouds'][closest_idx]['count'] - 1) + x) / \
                                                      self.state['clouds'][closest_idx]['count']
            self.state['clouds'][closest_idx]['sigma'] = (self.state['clouds'][closest_idx]['sigma'] *
                                                          (self.state['clouds'][closest_idx]['count'] - 1) +
                                                          np.linalg.norm(x) ** 2) / self.state['clouds'][closest_idx][
                                                             'count']
            self.state['clouds'][closest_idx]['count'] += 1

            # Adapt parameters
            self._adapt_parameters(closest_idx, x)

    def _adapt_parameters(self, cloud_idx, x):
        """
        Adapt controller parameters using tracking error
        :param cloud_idx: Index of cloud to update
        :param x: Normalized input vector
        """
        cloud = self.state['clouds'][cloud_idx]
        error = self.state['tracking_error']
        delta_error = error - self.state['prev_error']
        self.state['prev_error'] = error

        # Calculate adaptation terms
        adaptation = self.state['adapt_params']['gamma'] * error * x

        # Update PID parameters with leakage
        cloud['parameters']['P'] += adaptation[0] - self.state['adapt_params']['lambda'] * cloud['parameters']['P']
        cloud['parameters']['I'] += adaptation[1] - self.state['adapt_params']['lambda'] * cloud['parameters']['I']
        cloud['parameters']['D'] += adaptation[2] - self.state['adapt_params']['lambda'] * cloud['parameters']['D']
        cloud['parameters']['R'] += adaptation[3] - self.state['adapt_params']['lambda'] * cloud['parameters']['R']

        # Apply parameter limits
        cloud['parameters']['P'] = np.clip(cloud['parameters']['P'], 0, 1)
        cloud['parameters']['I'] = np.clip(cloud['parameters']['I'], 0, 0.1)
        cloud['parameters']['D'] = np.clip(cloud['parameters']['D'], 0, 0.05)
        cloud['parameters']['R'] = np.clip(cloud['parameters']['R'], -1, 1)

    def _fuzzy_inference(self, x):
        """
        Fuzzy inference system using cloud points
        :param x: Normalized input vector
        :return: Calculated control signal
        """
        total_weight = 0.0
        control_sum = 0.0

        for cloud in self.state['clouds']:
            # Calculate membership value
            distance = np.linalg.norm(x - cloud['mu'])
            membership = np.exp(-(distance ** 2) / (2 * cloud['sigma'] ** 2))

            # Calculate local control signal
            p = cloud['parameters']['P']
            i = cloud['parameters']['I']
            d = cloud['parameters']['D']
            r = cloud['parameters']['R']
            local_control = p * self.state['tracking_error'] + \
                            i * self.state['integral_sum'] + \
                            d * (self.state['tracking_error'] - self.state['prev_error']) + r

            # Update totals
            control_sum += membership * local_control
            total_weight += membership

        if total_weight == 0:
            return 0.0
        return control_sum / total_weight

    def _apply_protection(self, control):
        """
        Apply protection mechanisms to control signal
        :param control: Raw control signal
        :return: Protected control signal
        """
        # Dead zone
        if abs(self.state['tracking_error']) < self.dead_zone:
            return 0.0

        # Rate limiting
        if self.state.get('prev_control') is not None:
            control = np.clip(control,
                              self.state['prev_control'] - 10,
                              self.state['prev_control'] + 10)

        self.state['prev_control'] = control
        return control

    def reset(self):
        """
        Reset controller state
        """
        self.state = {
            'clouds': [],
            'adapt_params': {
                'P': [], 'I': [], 'D': [], 'R': [],
                'gamma': 0.1, 'lambda': 0.5, 'sigma': 0.1
            },
            'tracking_error': 0.0,
            'prev_error': 0.0,
            'integral_sum': 0.0,
            'ref_model_output': self.target
        }