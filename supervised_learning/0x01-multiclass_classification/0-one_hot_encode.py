#!/usr/bin/env python3

"""function converts a numeric label
    vector into a one-hot matrix"""

import numpy as np

def one_hot_encode(Y, classes):
    """returns one_hot matrix"""
    try:
        A =  np.eye(classes)[Y]
        return A.T
    except:
        return None




