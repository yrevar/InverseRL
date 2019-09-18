import numpy as np

# Adapted from https://github.com/david-abel/simple_rl/simple_rl/mdp/StateClass.py
class State(object):
    ''' State specification class '''

    def __init__(self, identifier=[]):
        self.identifier = identifier
        self.class_id = None
        self.features = None

    def get_id(self):
        return self.identifier

    def get_class(self):
        return self.class_id

    def get_features(self):
        return self.features

    def attach_class(self, class_id):
        self.class_id = class_id

    def attach_features(self, features):
        self.features = features

    def __hash__(self):
        if type(self.identifier).__module__ == np.__name__:
            # Numpy arrays
            return hash(str(self.identifier))
        elif self.identifier.__hash__ is None:
            return hash(tuple(self.identifier))
        else:
            return hash(self.identifier)

    def __str__(self):
        return "State: (" + str(self.identifier) + ")"

    def __eq__(self, other):
        if isinstance(other, State):
            return self.identifier == other.identifier
        return False

    def __getitem__(self, index):
        return self.identifier[index]

    def __len__(self):
        return len(self.identifier)