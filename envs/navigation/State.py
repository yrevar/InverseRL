import numpy as np

# Adapted from https://github.com/david-abel/simple_rl/simple_rl/mdp/StateClass.py
class State(object):
    ''' State specification class '''

    def __init__(self, location, idx=None,
                 class_id=None, features=None,
                 reward=None, terminal_status=False):
        self.location = location
        self.idx = idx
        self.class_id = class_id
        self.features = features
        self.reward = reward
        self.terminal_status = terminal_status

    def get_idx(self):
        return self.idx

    def get_id(self):
        return self.location

    def get_class(self):
        return self.class_id

    def get_features(self):
        return self.features

    def get_reward(self):
        return self.reward

    def is_terminal(self):
        return self.terminal_status

    def attach_class(self, class_id):
        self.class_id = class_id

    def attach_features(self, features):
        self.features = features

    def attach_reward(self, reward):
        self.reward = reward

    def set_terminal_status(self, b_terminal_status):
        self.terminal_status = b_terminal_status

    def __hash__(self):
        if type(self.location).__module__ == np.__name__:
            # Numpy arrays
            return hash(str(self.location))
        elif self.location.__hash__ is None:
            return hash(tuple(self.location))
        else:
            return hash(self.location)

    def __str__(self):
        return "State: " + str(self.location)

    def _meta(self):
        return "State: " + str(self.location) + " [ "+ \
               ("C {} ".format(self.get_class()) if self.get_class() is not None else "") + \
               ("R {:.2f} ".format(self.get_reward()) if self.get_reward() is not None else "") + \
               ("phi {} ".format(self.get_features().shape) if self.get_features() is not None else "") + \
               ("Terminal " if self.is_terminal() else "") + \
                "]"

    def __repr__(self):
        return str(self.__module__) + "." + self.__class__.__name__ + str(self.location) + " at " + hex(id(self))

    def __eq__(self, other):
        if isinstance(other, State):
            return self.location == other.location
        return False

    def __getitem__(self, index):
        return self.location[index]

    def __len__(self):
        return len(self.location)