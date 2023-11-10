from typing import List
from collections import defaultdict
import navigation_mdp as nvmdp
import utils
import numpy as np

from irl.data import TrajectoryStore
import rl.planning as Plan

from .AbstractGridWorldDemonstration import AbstractGridWorldDemonstration

class GridWorldDemonstration(AbstractGridWorldDemonstration):

    FEATURE_KEY_NONLINEAR = "nonlin_feature"

    def __init__(self, symbol_layout, symbol_to_class_id_dict, trajectories):
        super().__init__()
        self.initialize(symbol_layout, symbol_to_class_id_dict, trajectories)


    def initialize(self, symbol_layout, symbol_to_class_id_dict, trajectories):
        self.class_dist = nvmdp.class_.XYClassDistribution(symbol_layout, symbol_to_class_id_dict)
        self.class_ids = self.class_dist().flatten()
        self.h_cells, self.w_cells = self.class_dist().shape
        self.S = nvmdp.state.DiscreteStateSpace(self.h_cells, self.w_cells)
        self.S.attach_classes(self.class_ids)
        # self.S.attach_feature_spec(
        #     nvmdp.features.FeatureClassImageSampler(
        #         lambda class_id: utils.sample_image(self.x_train, self.y_train, class_id), key=self.FEATURE_KEY_IMAGE))
        self.T = nvmdp.dynamics.VonNeumannDynamics(self.S, slip_prob=0.0)
        self.trajectory_store = TrajectoryStore.discrete_compact_4_actions(
            trajectories, None
        )
        self.trajectory_store.infer_actions(lambda s_lst_lst: self._infer_actions(self.T, s_lst_lst))
        self.PHI = None
        self.PHI_gridded = None
        self.VI = defaultdict(lambda: None)

    def _infer_actions(self, dynamics, traj_list):
        a_lst_lst = []
        for traj in traj_list:
            a_lst_lst.append(dynamics.loc_lst_to_a_lst(traj))
        return a_lst_lst

    def get_states(self) -> nvmdp.state.DiscreteStateSpace:
        return self.S

    def attach_feature_spec(self, feature_spec):
        self.S.attach_feature_spec(feature_spec)

    def get_features(self, loc=None, idx=None, key=None) -> np.ndarray:
        if self.PHI is None:
            self.PHI = self.S.features(loc=loc, idx=idx, gridded=False, numpyize=True, key=key)
        return self.PHI

    def get_feature_shape(self, key=None):
        return self.get_features((0,0), key=key).shape

    def get_state_feature(self, s: nvmdp.state.State, key: str = None) -> np.ndarray:
        return s.get_features(key=key)

    def get_features_grid(self, key=None) -> np.ndarray:
        return self.S.features(gridded=True, key=key)

    def attach_reward_spec(self, reward_spec: nvmdp.reward.AbstractStateRewardSpec):
        self.R = reward_spec
        self.S.attach_reward_spec(self.R)

    def get_rewards(self, key=None):
        return self.S.rewards(numpyize=False, gridded=False, key=key)

    def get_actions(self) -> List:
        return self.T.actions()

    def get_transition_fn(self) -> nvmdp.dynamics.AbstractDynamics:
        return self.T

    def get_next_state(self, s, a) -> nvmdp.state.State:
        return self.T(s, a)

    def get_trajectories(self) -> TrajectoryStore:
        return self.trajectory_store

    def reshape_to_grid(self, values):
        return self.S._organize_to_grid(values)

    # def plan(self, goal, policy, vi_max_iters, reasoning_iters, vi_eps=1e-6, gamma=0.95,
    #          reward_key=None, verbose=False, debug=False):
    #     self.R_curr = self.get_rewards(key=reward_key)
    #     self.VI[goal] = Plan.ValueIteration(self.S, self.R_curr, self.T, verbose=verbose,
    #                                         log_pi=False, gamma=gamma, goal=goal)
    #     self.VI[goal].run(vi_max_iters, policy, reasoning_iters=reasoning_iters,
    #                       verbose=verbose, debug=debug, eps=vi_eps, ret_vals=False)
    #     return self.VI[goal]

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