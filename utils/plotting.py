import numpy as np
# Navigation Views
import navigation_vis.Raster as NavGridView
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import cm as cm, pyplot as plt, colors as mplotcolors


def plot_grid_data_helper(data, ann=None, ann_sz=12, ann_col="black", title=None, grid=None, cmap=cm.viridis):
    p = NavGridView.Raster(data, ax=plt.gca()).render(cmap=cmap).ticks(minor=False)
    if ann is not None:
        p.show_cell_text(ann, fontsize=ann_sz, color_cb=lambda x: ann_col)
    if grid is not None:
        p.grid()
    if title is not None:
        p.title(title)
    return p

def highlight_cell(x, y, ax=None, **kwargs):
    rect = plt.Rectangle((x-.5, y-.5), 1, 1, fill=False, **kwargs)
    ax = ax or plt.gca()
    ax.add_patch(rect)
    return rect

class NavGridViewPlotter:
    def __init__(self, S, R=None, cartesian=False):
        self.S = S
        self.R = S.rewards() if R is None else R
        self.PHI_gridded = self.S.features(gridded=True)
        self.R_grided = self.S._organize_to_grid(self.R)
        self.class_ids_grided = self.S._organize_to_grid(self.S.class_ids)
        self.idxs_gridded = self.S.idxs
        self.cartesian = cartesian
        self.p = None

    def highlight_terminal_states(self):
        for s in self.S.get_terminal_states():
            r, c = s.location
            highlight_cell(c, r, ax=self.p.ax, color="white", linewidth=5)

    def plot_rewards(self, title="Rewards", *args, **kwargs):
        data = self.R_grided[..., np.newaxis, np.newaxis, np.newaxis]
        ann = self.R
        self.p = plot_grid_data_helper(data, ann, title=title, *args, **kwargs)
        if self.cartesian:
            self.p.ax.invert_yaxis()
        return self.p

    def plot_states(self, title="States", *args, **kwargs):
        data = self.idxs_gridded[..., np.newaxis, np.newaxis, np.newaxis]
        ann = self.S.idxs.flatten()
        self.p = plot_grid_data_helper(data, ann, title=title, *args, **kwargs)
        self.highlight_terminal_states()
        if self.cartesian:
            self.p.ax.invert_yaxis()
        return self.p

    def plot_array(self, data, title="Data", *args, **kwargs):
        data = self.S._organize_to_grid(np.asarray(data).flatten())[..., np.newaxis, np.newaxis, np.newaxis]
        ann = data.flatten()
        self.p = plot_grid_data_helper(data, ann, title=title, *args, **kwargs)
        if self.cartesian:
            self.p.ax.invert_yaxis()
        return self.p

    def plot_classes(self, title="Classes", *args, **kwargs):
        data = self.class_ids_grided[..., np.newaxis, np.newaxis, np.newaxis]
        ann = self.S.class_ids
        self.p = plot_grid_data_helper(data, ann, title=title, *args, **kwargs)
        self.highlight_terminal_states()
        if self.cartesian:
            self.p.ax.invert_yaxis()
        return self.p

    def plot_features(self, ann=None, title="Features", *args, **kwargs):
        if self.cartesian:
            data = NavGridView.flip_y_axis(self.PHI_gridded)
        else:
            data = self.PHI_gridded
        n_dim = len(data.shape)
        if n_dim == 3: # one-hot features
            H, W, K = data.shape
            if K < 10:
                data = data[..., np.newaxis, np.newaxis]
            else:
                # k1 = int(np.ceil(np.sqrt(K)))
                # k2 = int(np.ceil(K / k1))
                try:
                    k1 = H
                    k2 = W
                    data = data.reshape(H, W, k1, k2)[..., np.newaxis]
                except:
                    data = data.reshape(H, W, K, 1)[..., np.newaxis]
        elif n_dim == 4:
            data = data[..., np.newaxis]
        elif n_dim == 5:
            pass
        else:
            raise Exception("data dimension {} not supported!".format(n_dim))

        if ann is None:
            ann = self.S.class_ids
        ann = np.asarray(ann).flatten()
        self.p = plot_grid_data_helper(data, ann, title=title, *args, **kwargs)
        if self.cartesian:
            self.p.ax.invert_yaxis()
        return self.p

    def add_colorbar(self, *args, **kwargs):
        self.p.colorbar(*args, **kwargs)
        return self

    def add_trajectories(self, *args, ** kwargs):
        self.p.add_trajectories(*args, **kwargs)
        return self
