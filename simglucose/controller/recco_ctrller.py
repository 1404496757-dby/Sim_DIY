from .base import Controller
from .base import Action
import logging

logger = logging.getLogger(__name__)


class PIDController(Controller):
    def __init__(self, target=140):
        self.target = target

    def policy(self, observation, reward, done, **kwargs):
        sample_time = kwargs.get('sample_time')

        # BG is the only state for this PID controller
        bg = observation.CGM
        insulin = bg/10

        logger.info('Control input: {}'.format(insulin))

        # return the action
        action = Action(basal=insulin, bolus=0)
        return action

    def reset(self):
        self.integrated_state = 0
        self.prev_state = 0
