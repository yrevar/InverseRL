import numpy as np
from envs.navigation.State import State
from utils.utils import one_hot_nd


class AbstractStateFeature:

    def __init__(self, state_space):
        self.state_space = state_space
        self.features_lst = None

    def __call__(self, loc=None, idx=None, gridded=False):
        if loc is not None:
            return self.get_features_at_loc(loc)
        elif idx is not None:
            return self.get_features_at_idx(idx)
        else:
            if gridded:
                return self.get_all_features().reshape(self.state_space.limits + self.features_lst.shape[1:])
            else:
                return self.get_all_features()

    def __getitem__(self, idx):
        return self.get_features_at_idx(idx)

    def __len__(self):
        return len(self.features_lst)

    def get_all_features(self):
        return self.features_lst

    def get_features_at_idx(self, idx):
        return self.features_lst[idx]

    def get_features_at_loc(self, loc):
        return self.features_lst[self.state_space.loc_to_idx_dict[loc]]


class FeatureStateIndicator(AbstractStateFeature):

    def __init__(self, state_space):
        super().__init__(state_space)
        self.features_lst = self.state_space.idxs.flatten()


class FeatureStateIndicatorOneHot(AbstractStateFeature):

    def __init__(self, state_space):
        super().__init__(state_space)
        self.features_lst = one_hot_nd(self.state_space.idxs.flatten(), N=self.state_space.n_states)


class FeatureClassIndicator(AbstractStateFeature):

    def __init__(self, state_space):
        super().__init__(state_space)
        self.features_lst = self.state_space.class_ids


class FeatureClassIndicatorOneHot(AbstractStateFeature):

    def __init__(self, state_space, K=None):
        super().__init__(state_space)
        max_class_id = max(self.state_space.class_ids) if K is None else K
        self.features_lst = one_hot_nd(self.state_space.class_ids, N=max_class_id + 1)


class FeatureClassImage(AbstractStateFeature):

    def __init__(self, state_space, feature_map):
        super().__init__(state_space)
        self.features_lst = np.asarray([feature_map[clsid] for clsid in self.state_space.class_ids.flatten()])


class FeatureClassImageSampler(AbstractStateFeature):

    def __init__(self, state_space, feature_sampler):
        super().__init__(state_space)
        self.features_lst = np.asarray([feature_sampler(clsid) for clsid in self.state_space.class_ids.flatten()])

