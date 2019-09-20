import itertools
import numpy as np
from envs.navigation.State import State

from utils.utils import one_hot, one_hot_nd


class DiscreteStateSpace:
    ''' Discrete State Space specification class '''

    def __init__(self, *args):
        self.n_dims = len(args)
        self.limits = args
        self.n_states = np.product(self.limits)
        self.idxs = self._get_idxs()
        self.space = self._get_space()
        self.states_lst, self.loc_to_state_dict, self.state_to_loc_dict = self._get_states()
        # self.idxs.flatten() # default attributes are state indices
        self.class_ids = np.asarray([0] * self.n_states) # default class 0 for all state
        self.n_classes = 1
        # self.features_lst = self.idxs
        # self.feature_dim = self.features_lst.shape[-1]

    def _get_idxs(self):
        return np.arange(self.n_states).reshape(self.limits)

    def _get_space(self):
        return np.array(list(itertools.product(*[np.arange(lim) for lim in self.limits]))).reshape(
            self.limits + (self.n_dims,))

    def _get_states(self):

        state_list = []
        loc_to_state_dict = {}
        state_to_loc_dict = {}
        for loc in list(itertools.product(*[np.arange(lim) for lim in self.limits])):
            state = State(location=loc)
            state_list.append(state)
            loc_to_state_dict[loc] = state
            state_to_loc_dict[state] = loc
        return state_list, loc_to_state_dict, state_to_loc_dict

    def __str__(self):
        #return "".join(str(s) if ((idx+1) % self.limits[-1] != 0) else str(s) + "\n" for idx, s in enumerate(self.states_lst))
        return str(self.space)

    def at_loc(self, loc):
        return self.loc_to_state_dict[loc]

    def __call__(self, idx=None):
        if idx is None:
            return self.states_lst
        else:
            return self.states_lst[idx]

    def __getitem__(self, idx):
        return self.states_lst[idx]

    def __len__(self):
        return self.n_states

    def all(self):
        return self.states_lst

    def attach_classes(self, class_ids=[]):
        if len(class_ids) != self.n_states:
            raise Exception("Require class id for each state!")
        self.n_classes = len(np.unique(class_ids))
        self.class_ids = np.asarray(class_ids)
        for idx, class_id in enumerate(class_ids):
            self.states_lst[idx].attach_class(class_id)

    def sample_and_attach_classes(self, class_ids=[], p_dist=None):
        S = class_ids
        if p_dist is None:
            p_dist = np.ones(len(S)) / len(S)
        if len(p_dist) != len(class_ids):
            raise Exception("class_ids and p_dist must have same length!")
        self.n_classes = len(class_ids)
        self.class_ids = np.random.choice(S, self.n_states, p=p_dist) #.reshape(self.limits)
        for idx, class_id in enumerate(self.class_ids):
            self.states_lst[idx].attach_class(class_id)

    def class_ids(self):
        return self.class_ids

    def attach_features(self, PHI, *args, **kwargs):
        self.PHI = PHI(self, *args, **kwargs)
        self.features_lst = self.PHI()
        self.feature_dim = self.features_lst.shape[1:]
        for idx, features in enumerate(self.features_lst):
            self.states_lst[idx].attach_features(features)

    def features(self):
        return self.features_lst