from typing import List

import navigation_mdp as nvmdp
import numpy as np

from .AbstractGridWorldDemonstration import AbstractGridWorldDemonstration


class MnistGridWorldDemonstration(AbstractGridWorldDemonstration, nvmdp.state.DiscreteStateSpace):

    def __init__(self, H, W):
        super().__init__(H, W)

    def get_states(self) -> nvmdp.state.DiscreteStateSpace:
        pass

    def get_features(self) -> np.ndarray:
        pass

    def get_actions(self) -> List:
        pass

    def get_transition_fn(self) -> nvmdp.dynamics.AbstractDynamics:
        pass

    def get_next_state(self, s, a) -> nvmdp.state.State:
        pass

    def get_trajectories(self) -> List:
        pass
