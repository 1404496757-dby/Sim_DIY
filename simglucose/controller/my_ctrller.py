# from .user_interface import simulate
from .base import Controller, Action


class MyController(Controller):
    def __init__(self, insulin=0):
        self.insulin = insulin #init_state
        # self.state = init_state
        # super().__init__(init_state)

    def policy(self, observation, reward, done, **info):
        '''
        Every controller must have this implementation!
        ----
        Inputs:
        observation - a namedtuple defined in simglucose.simulation.env. For
                      now, it only has one entry: blood glucose level measured
                      by CGM sensor.
        reward      - current reward returned by environment
        done        - True, game over. False, game continues
        info        - additional information as key word arguments,
                      simglucose.simulation.env.T1DSimEnv returns patient_name
                      and sample_time
        ----
        Output:
        action - a namedtuple defined at the beginning of this file. The
                 controller action contains two entries: basal, bolus
        '''
        # self.state = observation
        action = Action(basal=self.insulin, bolus=0)
        return action

    def update_insulin(self,get_insulin):
        self.insulin = get_insulin

    def reset(self):
        '''
        Reset the controller state to inital state, must be implemented
        '''
        # self.state = self.init_state
        pass


# ctrller = MyController(0)
# simulate(controller=ctrller)
