import numpy as np

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
