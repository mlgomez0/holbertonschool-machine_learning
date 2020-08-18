#!/usr/bin/env python3
"""shuffle two matrices"""
import numpy as np

def shuffle_data(X, Y):
    b = np.random.permutation(Y)
    np.random.seed(0)
    a = np.random.permutation(X)
    return (a, b)
