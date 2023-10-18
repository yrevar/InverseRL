import shutil
import time
import os, os.path as osp
import random, io, imageio
import numpy as np
from PIL import Image
from IPython import display
from collections import defaultdict, Counter
import cv2
import torch

import matplotlib.pyplot as plt


def plot_trajctory(traj, format="yx", **kwargs):
    traj = np.asarray(traj)
    if format == "yx":
        plt.plot(traj[:, 1], traj[:, 0], **kwargs)
    else:
        plt.plot(traj[:, 0], traj[:, 1], **kwargs)


def plot_img_with_trajectories(img, trajectories, im_kwargs={}, traj_kwargs={}):
    h, w, c = img.shape
    plt.imshow(img, **im_kwargs)
    for traj in trajectories:
        plot_trajctory(traj, **traj_kwargs)
    plt.xlim([0, w])
    plt.ylim([0, h])
    plt.gca().invert_yaxis()


def plot_img_with_trajectory(img, traj, im_kwargs={}, traj_kwargs={}):
    h, w, c = img.shape
    plt.imshow(img, **im_kwargs)
    plot_trajctory(traj, **traj_kwargs)
    plt.xlim([0, w])
    plt.ylim([0, h])
    plt.gca().invert_yaxis()


def convert_to_4_actions_path(path):
    x_, y_ = None, None
    new_path = []
    for x, y in path:
        if x_ is None and y_ is None:
            new_path.append([x, y])
        elif x_ != x and y_ != x:
            new_path.append([x, y_])
            new_path.append([x, y])
        else:
            new_path.append([x, y])
        x_, y_ = x, y
    return new_path


def compact_paths(paths, restrict_4_actions=False):
    """
    Removes None action. Set restrict_4_actions to True to remove other than cardinal directions.
    :param paths:
    :param restrict_4_actions:
    :return:
    """
    new_paths = []
    for path in paths:
        if restrict_4_actions:
            path = convert_to_4_actions_path(path)
        new_path = []
        s_old = None
        for s in path:
            if s_old is None or np.any(s != s_old):
                new_path.append(tuple(s))
                s_old = s
        new_paths.append(new_path)
    return new_paths


def normalize(np_array):
    normed = (np_array - np_array.min()) / (np_array.max() - np_array.min())
    normed = np.minimum(normed, 1)
    normed = np.maximum(normed, 0)
    return normed


def compute_epoch(batch_idx, batch_size, data_size):
    return int(np.floor(batch_idx * batch_size / data_size))


def get_lab_freq(labels, label_to_str=None, precision=4):
    l_cnts = dict(Counter(labels))
    s = sum(l_cnts.values())
    if s == 0:
        raise Exception("invalid input!")
    else:
        if label_to_str is not None:
            return {label_to_str[k]: round(v / s, precision) for k, v in l_cnts.items()}
        else:
            return {k: round(v / s, precision) for k, v in l_cnts.items()}


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
        return rand_idxs, X[rand_idxs].copy(), tuple(var[rand_idxs].copy() for var in vars)
    else:
        return rand_idxs, X[rand_idxs], tuple(var[rand_idxs] for var in vars)


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

    image_stack = np.reshape(image_stack.copy(), [-1, H, W, C])

    if nrow is None:
        nrow = int(np.ceil(np.sqrt(np.shape(image_stack)[0])))

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
                self.gif_writer = imageio.get_writer(fname, mode=mode, fps=fps)
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


def read_image(fname, resize_shape=None):
    if resize_shape is None:
        return cv2.imread(fname)[:, :, ::-1].copy()
    else:
        return cv2.resize(cv2.imread(fname)[:, :, ::-1].copy(), resize_shape)


def user_input(message, choices=("y", "n")):
    choice = input("%s (%s) " % (message, "/".join(choices)))
    if choice.strip() in choices:
        return choice.strip()
    else:
        raise ValueError("Invalid input!")


def make_clean_dir(dirpath, confirm_deletion=True):
    if osp.exists(dirpath):
        if not confirm_deletion or \
                "y" == user_input(
            "This will delete {}. Are you sure?".format(osp.abspath(dirpath)), choices=("y", "n")):
            print("Cleaning...\n\t{}".format(osp.abspath(dirpath)))
            shutil.rmtree(dirpath)
        else:
            print("Not cleaning...")
            return False
    os.makedirs(dirpath)
    return True


def log_likelihood(Pi, traj_list, eps=1e-15, ignore_last_action=True):
    loglik = 0
    for traj in traj_list:
        if ignore_last_action:
            traj = traj[:-1]
        for s, a in traj:
            loglik += torch.log(max(torch.tensor(eps), min(torch.tensor(1.) - eps, Pi[s, a])))
    return loglik


def plot_discrete_array(data, rep_axes=[1, 1], cmap_nm="tab20", interpolation="none"):
    for i, rep in enumerate(rep_axes):
        data = np.repeat(data, rep, axis=i)
    cmap = plt.get_cmap(cmap_nm, np.max(data) - np.min(data) + 1)
    mat = plt.imshow(data, cmap=cmap, interpolation=interpolation,
                     vmin=np.min(data) - .5, vmax=np.max(data) + .5)
    cax = plt.colorbar(mat, ticks=np.arange(np.min(data), np.max(data) + 1))


def cost_regularization_term(r_loc_fn, traj_list, ignore_last_action=True):
    reg = 0
    for traj in traj_list:
        if ignore_last_action:
            traj = traj[:-1]
        for s, a in traj:
            reg += r_loc_fn(s)
    return reg
