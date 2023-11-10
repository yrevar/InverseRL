import numpy as np
import os.path as osp
from typing import List
from collections import defaultdict
import utils
import navigation_mdp as nvmdp
import torch
from irl.data import TrajectoryStore
import rl.planning as Plan
import rl.policy as Policy
from sklearn.decomposition import PCA
from .AbstractGridWorldDemonstration import AbstractGridWorldDemonstration
from rl.model import ConvFCAutoEncoder, ConvAutoEncoder, RewardLinear


def image_preprocess_fn(x):
    if len(x.shape) == 3:
        return torch.FloatTensor(x).unsqueeze(0).permute(0, 3, 1, 2)
    else:
        return torch.FloatTensor(x).permute(0, 3, 1, 2)
    return x


class SensoryGridWorldDemonstration(AbstractGridWorldDemonstration):

    FEATURE_KEY_IMAGE = "feature_image"

    def __init__(self, map_fname, traj_fname, h_cells, w_cells, h_aug=0, w_aug=0, resize=None, traj_pre_discretized=False):
        super().__init__()
        self.map_img = utils.read_image(map_fname, resize) / 255.
        self.map_trajectories = self._read_trajectory_file(traj_fname)
        self.h_cells = h_cells
        self.w_cells = w_cells
        self.h_aug = h_aug
        self.w_aug = w_aug
        self.traj_pre_discretized = traj_pre_discretized
        self.PHI = None
        self.PHI_gridded = None
        self.r_spec = None
        self.initialize()

    def initialize(self):
        self.map_img_dtizer = nvmdp.features.ImageDiscretizer(self.map_img, self.h_cells, self.w_cells, (self.h_aug, self.w_aug))
        self.map_grid = self.map_img_dtizer()
        self.loc_to_array_fn = lambda row, col: self.map_img_dtizer.get_image_cell(row, col)
        self.S = nvmdp.state.DiscreteStateSpace(self.h_cells, self.w_cells)
        self.S.attach_classes(np.arange(self.h_cells * self.w_cells))
        self.S.attach_feature_spec(
            nvmdp.features.FeatureStateLocToArray(self.loc_to_array_fn, key=self.FEATURE_KEY_IMAGE))
        self.T = nvmdp.dynamics.VonNeumannDynamics(self.S, slip_prob=0.0)
        if self.traj_pre_discretized:
            self.trajectory_store = TrajectoryStore.discrete_compact_4_actions(
                self.map_trajectories, None
            )
        else:
            self.trajectory_store = TrajectoryStore.continuous_compact_4_actions(
                self.map_trajectories, None, lambda x: self.map_img_dtizer.transform_state_list_list(x)
            )
        self.trajectory_store.infer_actions(lambda s_lst_lst: self._infer_actions(self.T, s_lst_lst))
        self.VI = defaultdict(lambda: None)

    def attach_reward_spec(self, reward_spec: nvmdp.reward.AbstractStateRewardSpec):
        self.R = reward_spec
        self.S.attach_reward_spec(self.R)

    def _read_trajectory_file(self, fname):
        if osp.exists(fname):
            return utils.compact_paths(np.load(fname, allow_pickle=True))
        else:
            return []

    def _infer_actions(self, dynamics, traj_list):
        a_lst_lst = []
        for traj in traj_list:
            a_lst_lst.append(dynamics.loc_lst_to_a_lst(traj))
        return a_lst_lst

    def get_states(self) -> nvmdp.state.DiscreteStateSpace:
        return self.S

    def get_features(self, loc=None, idx=None, key=None) -> np.ndarray:
        if self.PHI is None:
            self.PHI = self.S.features(loc=loc, idx=idx, gridded=False, numpyize=True, key=key)
        return self.PHI

    def get_feature_shape(self, key=None):
        return self.get_features((0,0), key=key).shape

    def get_features_grid(self, key=None) -> np.ndarray:
        return self.S.features(gridded=True, key=key)

    def get_rewards(self, key=None):
        return self.S.rewards(numpyize=False, gridded=False, key=key)

    def reshape_to_grid(self, values):
        return self.S._organize_to_grid(values)

    def get_actions(self) -> List:
        return self.T.actions()

    def get_transition_fn(self) -> nvmdp.dynamics.AbstractDynamics:
        return self.T

    def get_next_state(self, s, a) -> nvmdp.state.State:
        return self.T(s, a)

    def _get_s_lst_lst(self, format="yx"):
        assert format in ["xy", "yx"]
        if format is "yx":
            return self.s_lst_lst
        else:
            return [[(p[1], p[0]) for p in s_lst] for s_lst in self.s_lst_lst]

    def get_trajectories(self):
        return self.trajectory_store

    def preheat_rewards(self, goal, reward_key, r_init=-0.1, r_goal_init=0., lr=1e-3, n_epochs=200):
        import torch
        loss_fn = torch.nn.MSELoss()
        s_to_idx = {v: k for k, v in enumerate(self.S)}
        r_expected = torch.tensor(np.ones(len(self.S)) * r_init, dtype=torch.float32)
        r_expected[s_to_idx[self.S.at_loc(goal)]] = r_goal_init
        r_model = self.S.get_reward_spec(key=reward_key).get_model()
        old_lr = r_model.optimizer.param_groups[0]['lr']
        # set new learning rate
        for p in r_model.optimizer.param_groups:
            p['lr'] = lr
        print("Pre-heating rewards...")
        for epoch in range(n_epochs):
            r_model.zero_grad()
            rewards = torch.stack(self.get_rewards(key=reward_key))
            loss = loss_fn(rewards, r_expected)
            print(loss.detach().numpy().round(5), end=", ")
            loss.backward()
            r_model.step()
            if loss < 1e-4:
                print("Good enough!")
                break
        r_model.zero_grad()
        # reset old learning rate
        for p in r_model.optimizer.param_groups:
            p['lr'] = old_lr
        print(lr, old_lr)
        print("Ready to bake!")

    def plan(self, goal, policy, vi_max_iters, reasoning_iters, vi_eps=1e-6, gamma=0.95,
             reward_key=None, preheat_rewards=False, verbose=False, debug=False):
        if preheat_rewards:
            self.preheat_rewards(goal, reward_key)
        self.R_curr = self.get_rewards(key=reward_key)
        self.VI[goal] = Plan.ValueIteration(self.S, self.R_curr, self.T, verbose=verbose,
                                            log_pi=False, gamma=gamma, goal=goal)
        self.VI[goal].run(vi_max_iters, policy, reasoning_iters=reasoning_iters,
                          verbose=verbose, debug=debug, eps=vi_eps, ret_vals=False)
        return self.VI[goal]