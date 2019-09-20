import numpy as np

class AbstractStateRewardSpec:

    def __init__(self, state_space):
        self.state_space = state_space
        pass

    def of(self, state):
        return state.reward

    def __call__(self, state):
        return self.of(state)

class StateRewardScalar(AbstractStateRewardSpec):

    def __init__(self):
        pass