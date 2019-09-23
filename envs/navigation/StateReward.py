import numpy as np

class AbstractStateRewardSpec:

    def __init__(self, state_space):
        self.state_space = state_space
        self.reward_lst = None

    def __call__(self, idx=None):
        if idx is None:
            return self.get_reward_lst()
        else:
            return self.get_reward(idx)

    def __getitem__(self, idx):
        return self.reward_lst[idx]

    def __len__(self):
        return len(self.reward_lst)

    def get_reward_lst(self):
        return self.reward_lst

    def get_reward(self, idx):
        return self.reward_lst[idx]

class RewardStateScalar(AbstractStateRewardSpec):

    def __init__(self, state_space, loc_to_reward_dict, class_id_to_reward_dict, default=0):
        super().__init__(state_space)
        self.loc_to_reward_dict = loc_to_reward_dict
        self.class_id_to_reward_dict = class_id_to_reward_dict
        self.reward_lst = []
        for state in self.state_space.states_lst:
            # loc_to_reward_dict overrides class_id_to_reward_dict
            if (loc_to_reward_dict is not None) and state.location in loc_to_reward_dict:
                self.reward_lst.append(loc_to_reward_dict[state.location])
            elif (class_id_to_reward_dict is not None) and state.class_id in class_id_to_reward_dict:
                self.reward_lst.append(class_id_to_reward_dict[state.class_id])
            else:
                self.reward_lst.append(default)
        self.reward_lst = np.asarray(self.reward_lst)