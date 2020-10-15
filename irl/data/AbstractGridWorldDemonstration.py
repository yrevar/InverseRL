from abc import ABC, abstractmethod
import numpy as np
import navigation_mdp as nvmdp
from typing import Any, List, Dict, Union, Optional


class AbstractGridWorldDemonstration(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def get_states(self) -> nvmdp.state.DiscreteStateSpace:
        pass

    def get_state_feature(self, s: nvmdp.state.State) -> np.ndarray:
        pass

    @abstractmethod
    def get_features(self) -> np.ndarray:
        pass

    # @abstractmethod
    # def feature_pre_process(self, x) -> np.ndarray:
    #     pass
    #
    # @abstractmethod
    # def feature_post_process(self) -> np.ndarray:
    #     pass

    @abstractmethod
    def get_actions(self) -> List:
        pass

    @abstractmethod
    def get_transition_fn(self) -> nvmdp.dynamics.AbstractDynamics:
        pass

    @abstractmethod
    def get_next_state(self, s, a) -> nvmdp.state.State:
        pass

    @abstractmethod
    def get_trajectories(self) -> List:
        pass

    @abstractmethod
    def reshape_to_grid(self, values):
        pass
