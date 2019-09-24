import numpy as np

class AbstractDynamics:

    def __init__(self, state_space):
        self.state_space = state_space
        pass

    def take_action(self, state, action):
        raise NotImplementedError

    def get_next_states_distribution(self, state, action):
        raise NotImplementedError

    def __call__(self, state, action):
        return self.get_next_states_distribution(state, action)


class XYDynamics(AbstractDynamics):
    ACTIONS = ["U", "D", "L", "R"]
    OOPS_ACTIONS = {"U": ["L", "R"], "D": ["R", "L"], "L": ["D", "U"], "R": ["U", "D"]}

    def __init__(self, state_space, slip_prob=0.):
        super().__init__(state_space)
        self.slip_prob = slip_prob
        self.H, self.W = self.state_space.limits

    def _next_state(self, state, action):
        loc = state.location
        if action == "U" and loc[1] + 1 < self.H:
            loc_prime = (loc[0], loc[1] + 1)
        elif action == "D" and loc[1] - 1 > 0:
            loc_prime = (loc[0], loc[1] - 1)
        elif action == "L" and loc[0] - 1 > 0:
            loc_prime = (loc[0] - 1, loc[1])
        elif action == "R" and loc[0] + 1 < self.W:
            loc_prime = (loc[0] + 1, loc[1])
        else:
            if action not in self.ACTIONS:
                raise Exception("Invalid action {}!".format(action))
            else:
                loc_prime = loc
        # print(state, action, "->", loc_prime)
        return self.state_space.loc_to_state_dict[loc_prime]

    def get_next_states_distribution(self, state, action):
        action_list = [action, ] + self.OOPS_ACTIONS[action]
        p_vals = [ 1. - self.slip_prob, ] + [ self.slip_prob ] * len(self.OOPS_ACTIONS[action])
        return [(self._next_state(state, action), p_vals[idx]) for idx, action in enumerate(action_list)]

    def take_action(self, state, action):
        if state.is_terminal():
            return state
        if self.slip_prob > np.random.random():
            action = np.random.choice(OOPS_ACTIONS[action])
        return self._next_state(state, action)