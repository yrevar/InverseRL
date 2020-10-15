import numpy as np
import os.path as osp
from typing import List
import utils
import navigation_mdp as nvmdp
import torch

from .AbstractGridWorldDemonstration import AbstractGridWorldDemonstration

class SatelliteGridWorldDemonstration(AbstractGridWorldDemonstration):

    def __init__(self, map_fname, traj_fname, h_cells, w_cells, h_aug=0, w_aug=0):
        super().__init__()
        self.map_img = utils.read_image(map_fname)
        self.map_trajectories = self._read_trajectory_file(traj_fname)
        self.h_cells = h_cells
        self.w_cells = w_cells
        self.h_aug = h_aug
        self.w_aug = w_aug
        self.initialize()

    def initialize(self):
        self.map_img_dtizer = nvmdp.features.ImageDiscretizer(self.map_img, self.h_cells, self.w_cells, (self.h_aug, self.w_aug))
        self.map_grid = self.map_img_dtizer()
        self.loc_to_array_fn = lambda row, col: self.map_img_dtizer.get_image_cell(row, col)
        self.S = nvmdp.state.DiscreteStateSpace(self.h_cells, self.w_cells)
        self.S.attach_feature_spec(nvmdp.features.FeatureStateLocToArray(self.loc_to_array_fn, key="sat_image"))
        self.S.attach_classes(np.arange(self.h_cells * self.w_cells))
        self.T = nvmdp.dynamics.XYDynamics(self.S, slip_prob=0.0)
        self.s_lst_lst = utils.compact_paths(
            self.map_img_dtizer.transform_state_list_list(self.map_trajectories), restrict_4_actions=True)
        self.a_lst_lst = self._infer_actions(self.T, self.s_lst_lst)

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

    def get_features(self, loc=None, idx=None) -> np.ndarray:
        return self.S.features(loc=loc, idx=idx, gridded=False, numpyize=True, key="sat_image")

    def reshape_to_grid(self, values):
        return self.S._organize_to_grid(values)

    def get_actions(self) -> List:
        return self.T.actions()

    def get_transition_fn(self) -> nvmdp.dynamics.AbstractDynamics:
        return self.T

    def get_next_state(self, s, a) -> nvmdp.state.State:
        return self.T(s, a)

    def get_trajectories(self, states_only=False, s_a_zipped=False) -> List:
        if states_only:
            return self.s_lst_lst
        else:
            if s_a_zipped:
                return [list(zip(self.s_lst_lst[i], self.a_lst_lst[i])) for i in range(len(self.s_lst_lst))]
            else:
                return self.s_lst_lst, self.a_lst_lst
