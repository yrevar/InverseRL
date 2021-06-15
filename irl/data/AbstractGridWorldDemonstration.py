from abc import ABC, abstractmethod
import numpy as np
import navigation_mdp as nvmdp
from typing import Any, List, Dict, Union, Optional
from irl.data import TrajectoryStore


class AbstractGridWorldDemonstration(ABC):

    def __init__(self):
        return

    @abstractmethod
    def get_states(self) -> nvmdp.state.DiscreteStateSpace:
        return

    @abstractmethod
    def get_state_feature(self, s: nvmdp.state.State) -> np.ndarray:
        return

    @abstractmethod
    def get_features(self) -> np.ndarray:
        return

    @abstractmethod
    def get_features_grid(self) -> np.ndarray:
        return

    @abstractmethod
    def attach_reward_spec(self, reward_spec: nvmdp.reward.AbstractStateRewardSpec):
        return

    @abstractmethod
    def get_rewards(self):
        return

    @abstractmethod
    def get_actions(self) -> List:
        return

    @abstractmethod
    def get_transition_fn(self) -> nvmdp.dynamics.AbstractDynamics:
        return

    @abstractmethod
    def get_next_state(self, s, a) -> nvmdp.state.State:
        return

    @abstractmethod
    def get_trajectories(self) -> TrajectoryStore:
        return

    @abstractmethod
    def reshape_to_grid(self, values):
        return

    @abstractmethod
    def plan(self, *args, **kwargs):
        return


