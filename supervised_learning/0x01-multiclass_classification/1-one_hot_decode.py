#!/usr/bin/env python3

"""function converts a One-Hot matrix
    into vector of labels"""

import numpy as np


def one_hot_decode(one_hot):
    """returns vector of labels"""
    if len(one_hot) == 0:
        return None
    if np.sum((np.sum(one_hot, axis=0))) != one_hot.shape[1]:
        return None
    return np.argmax(one_hot, axis=0)
