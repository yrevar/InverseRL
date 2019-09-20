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
        self.class_ids = self.idxs # default attributes are state indices
        self.n_classes = self.n_states
        self.features_lst = self.idxs
        self.feature_dim = self.features_lst.shape[-1]

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

    def get_states_grid(self):
        return np.asarray(self.states_lst).reshape(self.limits)

    def attach_classes(self, class_ids=[], p_dist=None):
        S = class_ids
        if p_dist is None:
            p_dist = np.ones(len(S)) / len(S)
        if len(p_dist) != len(class_ids):
            raise Exception("class_ids and p_dist must have same length!")
        self.n_classes = len(class_ids)
        self.class_ids = np.random.choice(S, self.n_states, p=p_dist) #.reshape(self.limits)
        for idx, class_id in enumerate(self.class_ids):
            self.states_lst[idx].attach_class(class_id)

    def get_class_dist(self): # returns what generally referred to as states
        return self.class_ids

    def attach_features(self, kind, attrib_to_feature_map=None, feature_sampler=None):
        if kind == "state":
            self.features_lst = self.space
        elif kind == "state_idx":
            self.features_lst = self.idxs
        elif kind == "state_idx_oh": # orthogonal basis
            self.features_lst = one_hot_nd(self.idxs, N=self.n_states)
        elif kind == "attrib_idx":
            self.features_lst = self.class_ids
        elif kind == "attrib_idx_oh": # orthogonal basis
            self.features_lst = one_hot_nd(self.class_ids, N=self.n_classes)
        elif kind == "attrib_to_feature_map":
            if attrib_to_feature_map is None:
                raise Exception("Attribute to feature map dict must be specified for \"kind={}\"!".format(kind))
            self.features_lst = np.asarray([attrib_to_feature_map[a] for a in self.class_ids.flatten()])
        elif kind == "attrib_to_feature_sample":
            if feature_sampler is None:
                raise Exception("Feature sampler fn must be specified for \"kind={}\"!".format(kind))
            self.features_lst = np.asarray([feature_sampler(a) for a in self.class_ids.flatten()])
        #self.features_lst = self.features_lst.reshape(tuple(self.limits) + (-1,))
        # if self.features_lst.shape[len(self.idxs.shape):]:
        #     self.feature_dim = self.features_lst.shape[len(self.idxs.shape):]
        # else:
        #     self.feature_dim = 1
        self.feature_dim = self.features_lst.shape[1:]
        for idx, features in enumerate(self.features_lst):
            self.states_lst[idx].attach_features(features)