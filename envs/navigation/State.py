import numpy as np

# Adapted from https://github.com/david-abel/simple_rl/simple_rl/mdp/StateClass.py
class State(object):
    ''' State specification class '''

    def __init__(self, location=[]):
        self.location = location
        self.class_id = None
        self.features = None

    def get_id(self):
        return self.location

    def get_class(self):
        return self.class_id

    def get_features(self):
        return self.features

    def attach_class(self, class_id):
        self.class_id = class_id

    def attach_features(self, features):
        self.features = features

    def __hash__(self):
        if type(self.location).__module__ == np.__name__:
            # Numpy arrays
            return hash(str(self.location))
        elif self.location.__hash__ is None:
            return hash(tuple(self.location))
        else:
            return hash(self.location)

    def __str__(self):
        return "State: (" + str(self.location) + ")"

    def __eq__(self, other):
        if isinstance(other, State):
            return self.location == other.location
        return False

    def __getitem__(self, index):
        return self.location[index]

    def __len__(self):
        return len(self.location)