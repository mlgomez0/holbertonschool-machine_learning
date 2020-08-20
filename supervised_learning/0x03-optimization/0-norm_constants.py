#!/usr/bin/env python3
"""calculates normalizacion constants"""
import numpy as np

def normalization_constants(X):
    """return normalization constants"""
    m = X.mean(axis=0)
    s = X.std(axis=0)
    return m, s
