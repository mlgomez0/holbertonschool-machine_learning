#!/usr/bin/env python3
"""concatenates matrices along an specific axis"""
import numpy as np


def np_cat(mat1, mat2, axis=0):
    """returns new concated array"""
    return (np.concatenate((mat1, mat2), axis=axis))
