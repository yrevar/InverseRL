import time # hurray
import random
import numpy as np
from collections import defaultdict, Counter

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

