#!/usr/bin/env python3
"""calculates sensitivity for each class
   in a confusion matrix"""
import numpy as np


def sensitivity(confusion):
    """return sensitivity for each class"""
    return np.sum((confusion * np.identity(
                   confusion.shape[0]))/np.sum(confusion, axis=1), axis=1)
