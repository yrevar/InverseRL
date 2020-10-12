from abc import ABC, abstractmethod
import numpy as np
import navigation_mdp as NvMDP
from typing import Any, List, Dict, Union, Optional

class GridWorldDemonstrationDataset(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def get_states(self) -> NvMDP.state.DiscreteStateSpace:
        pass

    def get_state_feature(self, s: NvMDP.state.State) -> np.ndarray:
        pass

    @abstractmethod
    def get_features(self) -> np.ndarray:
        pass

    @abstractmethod
    def get_actions(self) -> List:
        pass

    @abstractmethod
    def get_transition_fn(self) -> NvMDP.dynamics.AbstractDynamics:
        pass

    @abstractmethod
    def get_next_state(self, s, a) -> NvMDP.state.State:
        pass

    @abstractmethod
    def get_trajectories(self) -> List:
        pass