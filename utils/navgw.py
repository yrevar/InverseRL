import torch
import navigation_mdp as NvMDP
from utils.plotting import NavGridViewPlotter as NvPlotter, plot_grid_data_helper
from matplotlib import cm as cm, pyplot as plt, colors as mplotcolors

def plot_navigation_world(S, s_lst_lst, rewards, titles=["States", "Classes", "Features", "Rewards"],
                          figsize=(18, 12), cbar_pad=1.0, cbar_size="10%"):
    plt.figure(figsize=figsize)
    plt.subplot(2, 2, 1)
    NvPlotter(S).plot_states(
        cmap=cm.viridis, ann_col="white",
        title=titles[0]).colorbar(where="left", pad=cbar_pad, size=cbar_size).grid().add_pixel_trajectories(
        [[(s[1], s[0]) for s in s_lst] for s_lst in s_lst_lst],
        arrow_props={"lw": 3, "color": "black", "shrinkB": 10, "shrinkA": 10})
    plt.subplot(2, 2, 2)
    NvPlotter(S).plot_classes(
        cmap=cm.viridis, ann_col="white",
        title=titles[1]).colorbar(where="right", pad=cbar_pad, size=cbar_size).grid()
    plt.subplot(2, 2, 3)
    NvPlotter(S).plot_features(
        ann=S.idxs.flatten(), cmap=None, ann_col="white",
        title=titles[2]).colorbar(where="left", pad=cbar_pad, size=cbar_size).grid().add_trajectories(
        [[(s[1], s[0]) for s in s_lst] for s_lst in s_lst_lst],
        arrow_props={"lw": 3, "color": "black", "shrinkB": 10, "shrinkA": 10})
    plt.subplot(2, 2, 4)
    NvPlotter(S).plot_array(
        rewards,
        cmap=cm.Blues_r, title=titles[3]).colorbar(where="right", pad=cbar_pad, size=cbar_size)
    plt.tight_layout()


def plot_irl_results(S, s_lst_lst, rewards, values, loglik_hist,
                     titles=["States", "Features", "Rewards", "Values", "Training Performance"],
                     figsize=(24, 24), cbar_pad=1.0, cbar_size="10%"):
    plt.figure(figsize=figsize)
    plt.subplot(3, 2, 1)
    NvPlotter(S).plot_states(
        cmap=cm.viridis, ann_col="white",
        title=titles[0]).colorbar(where="left", pad=cbar_pad, size=cbar_size).grid().add_pixel_trajectories(
        [[(s[1], s[0]) for s in s_lst] for s_lst in s_lst_lst],
        arrow_props={"lw": 3, "color": "black", "shrinkB": 10, "shrinkA": 10})
    plt.subplot(3, 2, 2)
    NvPlotter(S).plot_features(
        ann=S.idxs.flatten(), cmap=cm.viridis, ann_col="white",
        title=titles[1]).colorbar(where="right", pad=cbar_pad, size=cbar_size).grid().add_trajectories(
        [[(s[1], s[0]) for s in s_lst] for s_lst in s_lst_lst],
        arrow_props={"lw": 3, "color": "black", "shrinkB": 10, "shrinkA": 10})
    plt.subplot(3, 2, 3)
    NvPlotter(S).plot_array(
        rewards,
        cmap=cm.Blues_r, title=titles[2]).colorbar(where="left", pad=cbar_pad, size=cbar_size).add_pixel_trajectories(
        [[(s[1], s[0]) for s in s_lst] for s_lst in s_lst_lst],
        arrow_props={"lw": 3, "color": "black", "shrinkB": 10, "shrinkA": 10})
    plt.subplot(3, 2, 4)
    NvPlotter(S).plot_array(values, cmap=cm.Blues_r, title=titles[3]).colorbar(where="right", pad=cbar_pad,
                                                                               size=cbar_size)
    plt.subplot(3, 2, 5)
    plt.plot(list(range(len(loglik_hist))), loglik_hist)
    plt.xlabel("Iteration")
    plt.ylabel("Likelihood")
    plt.title(titles[4])
    plt.tight_layout()


class NavigationGridWorld:

    def __init__(self, H, W, xyclass_dist=None, terminal_loc_lst=[]):
        self.H, self.W = H, W
        self.clear_trajectories()
        self.S = NvMDP.state.DiscreteStateSpace(H, W)
        self.S.set_terminal_status_by_loc(terminal_loc_lst, b_terminal_status=True)
        self.T = NvMDP.dynamics.XYDynamics(self.S, slip_prob=0.)
        if xyclass_dist is not None:
            self.class_ids = NvMDP.class_.XYClassDistribution(xyclass_dist)().flatten()
            self.S.attach_classes(self.class_ids)

    def clear_trajectories(self):
        self.s_lst_lst = []
        self.a_lst_lst = []

    def add_trajectory(self, s_lst):
        self.s_lst_lst.append(s_lst)
        self.a_lst_lst.append(self.T.loc_lst_to_a_lst(s_lst))
        self.tau_lst = [list(zip(self.s_lst_lst[i], self.a_lst_lst[i])) for i in range(len(self.s_lst_lst))]

    def add_trajectories(self, s_lst_lst):
        for s_lst in s_lst_lst:
            self.s_lst_lst.append(s_lst)
            self.a_lst_lst.append(self.T.loc_lst_to_a_lst(s_lst))
        self.tau_lst = [list(zip(self.s_lst_lst[i], self.a_lst_lst[i])) for i in range(len(self.s_lst_lst))]

    def attach_features(self, loc_to_array_fn, feature_type="raw"):
        self.loc_to_array_fn = loc_to_array_fn
        self.S.attach_features(NvMDP.features.FeatureStateLocToArray(self.S, loc_to_array_fn), feature_type=feature_type)

    def attach_onehot_features(self, feature_type="raw"):
        self.S.attach_features(NvMDP.features.FeatureStateIndicatorOneHot(self.S), feature_type=feature_type)

    def attach_rewards(self, r_model, feature_type="raw"):
        self.r_model = r_model
        return self.S.attach_reward_model(lambda x: r_model(self.prepare_model_input(x)),
                                          feature_type=feature_type)  # H x W x C to C x H x W

    def features(self):
        return self.prepare_model_input(self.S.features())

    def dynamics(self):
        return self.T

    def demonstrations(self):
        return self.tau_lst

    def prepare_model_input(self, x):
        if not isinstance(x, torch.FloatTensor):
            x = torch.FloatTensor(x)

        if len(x.shape) == 3:
            return x.permute(2, 0, 1).unsqueeze(0)
        elif len(x.shape) == 4:
            return x.permute(0, 3, 1, 2)
        if len(x.shape) <= 2:
            return x
        else:
            raise ValueError("invalid input dim!")

    def get_feature_shape(self):
        return self.prepare_model_input(self.S[0].get_features()).shape

    def rewards(self):
        return self.S.rewards(numpyize=False)

    def trajectories(self):
        return self.s_lst_lst, self.a_lst_lst

    def plot_world(self, *args, **kwargs):
        plot_navigation_world(self.S, self.s_lst_lst, self.S.rewards(numpyize=True).round(3), *args, **kwargs)

    def plot_results(self, values, loglik_hist, *args, **kwargs):
        plot_irl_results(self.S, self.s_lst_lst, self.S.rewards(numpyize=True).round(3), values, loglik_hist,
                         *args, **kwargs)