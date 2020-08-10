#!/usr/bin/env python3

"""function converts a numeric label
    vector into a one-hot matrix"""

import numpy as np

def one_hot_encode(Y, classes):
    """returns one_hot matrix"""
    return np.eye(classes, Y.shape[0])[Y].T




