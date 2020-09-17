import time # hurray
import os.path as osp
import random, io, imageio
import numpy as np
from PIL import Image
from IPython import display
from collections import defaultdict, Counter

import matplotlib.pyplot as plt

def normalize(np_array):
    normed = (np_array - np_array.min()) / (np_array.max() - np_array.min())
    normed = np.minimum(normed, 1)
    normed = np.maximum(normed, 0)
    return normed

def compute_epoch(batch_idx, batch_size, data_size):
    return int(np.floor(batch_idx * batch_size/ data_size))

def get_lab_freq(labels, label_to_str=None, precision=4):

    l_cnts = dict(Counter(labels))
    s = sum(l_cnts.values())
    if s == 0:
        raise Exception("invalid input!")
    else:
        if label_to_str is not None:
            return {label_to_str[k]: round(v/s, precision) for k, v in l_cnts.items()}
        else:
            return {k: round(v/s, precision) for k, v in l_cnts.items()}

def calc_time_delta(func):
    def inner(*args, **kwargs):
        begin = time.time()
        func(*args, **kwargs)
        end = time.time()
        return end - begin
    return inner

def one_hot(i, n):
    v = np.zeros(n)
    v[i] = 1.
    return v

def one_hot_nd(nd_int_array, N=None):

    if N is None:
        N = len(np.unique(nd_int_array))

    oh_mat = []
    for x in np.nditer(nd_int_array):
        oh_mat.append(one_hot(x, N))
    return np.asarray(oh_mat).reshape(nd_int_array.shape + (N,))

def sample_image(X, y, y_q):
    x_q = X[y == y_q]
    return x_q[np.random.randint(0, len(x_q))]

def shuffle_in_sync(X, *vars, deep_copy=False):
    rand_idxs = np.random.choice(len(X), len(X), replace=False)
    if deep_copy:
        return X[rand_idxs].copy(), (*[var[rand_idxs].copy() for var in vars])
    else:
        return rand_idxs, X[rand_idxs], (*[var[rand_idxs] for var in vars])

def image_stack_to_sprite_image(image_stack, img_dim, nrow=None, padding=0):
    """
    Creates image grid of size nrow x (N/nrows) from N stacked images.

    :param img_stack (numpy array): a set of images in (N x H x W x C) format
    :param nrow: number of rows in grid
    :returns: image sprite with set of images laid out in a grid
    """
    H, W, C = img_dim
    import torch
    from torchvision import utils

    image_stack = np.reshape( image_stack.copy(), [ -1, H, W, C] )

    if nrow is None:
        nrow = int( np.ceil( np.sqrt( np.shape( image_stack )[ 0 ] ) ) )

    return utils.make_grid(
        torch.tensor(image_stack).permute(0, 3, 1, 2), nrow=nrow, padding=padding).permute(1, 2, 0)


def set_global_seeds(seed):
    try:
        import tensorflow as tf
    except ImportError:
        pass
    else:
        with tf.get_default_graph().as_default():
            tf.set_random_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def read_pil_image_from_plt():
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    return Image.open(buf)

class GifMaker:
    def __init__(self, fname=None, fps=10, live_view=False, mode="I"):
        self.live_view = live_view
        if fname is not None:
            if osp.exists(fname):
                ans = input("Gif file exists. Replace? y/n")
            if ans in ["y", "Y"]:
                self.gif_writer = imageio.get_writer(fname, mode=mode, fps=fps)
            else:
                self.gif_writer = None
        else:
            self.gif_writer = None

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.done()

    def add_plot(self):
        self.add_image(read_pil_image_from_plt())

    def add_image(self, img, show=False):
        if self.gif_writer is not None:
            self.gif_writer.append_data(np.array(img))
        if self.live_view:
            if show: plt.imshow(img)
            display.clear_output(wait=True)
            display.display(plt.gcf())

    def done(self):
        if self.gif_writer is not None:
            self.gif_writer.close()
        if self.live_view:
            plt.clf()

def get_xyz(img_2d):
    x_list, y_list, z_list = [], [], []
    for row in range(len(img_2d)):
        for col in range(len(img_2d[0])):
            x, y = row, col
            x_list.append(x)
            y_list.append(y)
            z_list.append(img_2d[row, col])
    return x_list, y_list, z_list

def plot_3d_barplot(img_2d, fig, elev=25, azim=-110, vmin=0, vmax=1, cmap=plt.cm.jet):
    X, Y, Z = get_xyz(img_2d)
    ax = fig.gca(projection='3d')
    top = Z
    bottom = np.zeros_like(top)
    width = depth = 1
    surf = ax.bar3d(Y, X, bottom, width, depth, top, shade=True)
    ax.set_zlim(0, 1.)
    ax.view_init(elev=elev, azim=azim)
    fig.colorbar(surf, shrink=0.5, aspect=5)

def plot_3d_surface(img_2d, fig, elev=25, azim=-110, vmin=0, vmax=1, cmap=plt.cm.jet):
    X, Y, Z = get_xyz(img_2d)
    ax = fig.gca(projection='3d')
    surf = ax.plot_trisurf(Y, X, Z, cmap=plt.cm.jet, linewidth=0.2, vmin=vmin, vmax=vmax)
    ax.set_zlim(0, 1.)
    ax.view_init(elev=elev, azim=azim)
    fig.colorbar(surf, shrink=0.5, aspect=5)
