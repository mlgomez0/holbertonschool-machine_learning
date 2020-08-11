#!/usr/bin/env python3

"""function converts a One-Hot matrix
    into vector of labels"""

import numpy as np

def one_hot_decode(one_hot):
    """returns vector of labels"""
    try:
        return np.argmax(one_hot, axis=0)
    except:
        return None



