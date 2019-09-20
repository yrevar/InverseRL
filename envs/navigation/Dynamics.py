import numpy as np

class Dynamics:

    def __init__(self, state_space):
        self.state_space = state_space
        pass

    def get_func(self):
        raise NotImplementedError

    def tick(self, state, action):
        raise NotImplementedError


class XYDynamics(Dynamics):
    ACTIONS = ["U", "D", "L", "R"]

    def __init__(self, state_space):
        super().__init__(state_space)
        self.H, self.W = self.state_space.limits

    def get_func(self):
        raise NotImplementedError

    def transition(self, state, action):

        loc = state.location
        if action == "U" and loc[1] + 1 < self.H:
            loc_prime = (loc[0], loc[1] + 1)
        elif action == "D" and loc[1] - 1 > 0:
            loc_prime = (loc[0], loc[1] - 1)
        elif action == "L" and loc[0] - 1 > 0:
            loc_prime = (loc[0] - 1, loc[1])
        elif action == "R" and loc[0] - 1 < self.W:
            loc_prime = (loc[0] + 1, loc[1])
        else:
            if action not in self.ACTIONS:
                raise Exception("Invalid action {}!".format(action))
            else:
                loc_prime = loc
        return self.state_space.loc_to_state_dict[loc_prime]