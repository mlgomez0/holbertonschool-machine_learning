#!/usr/bin/env python3
"""calculates normalizacion constants"""

def normalization_constants(X):
    m = X.mean(axis=0)
    s = X.std(axis=0)
    return (m, s)
