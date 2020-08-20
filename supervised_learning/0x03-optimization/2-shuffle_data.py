#!/usr/bin/env python3
"""shuffle two matrices"""
import numpy as np


def shuffle_data(X, Y):
    """returns shuffle matrix"""
    perm = X.shape[0]
    shuff_op = np.random.permutation(perm)
    return X[shuff_op], Y[shuff_op]
