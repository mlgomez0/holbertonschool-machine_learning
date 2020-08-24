#!/usr/bin/env python3
"""calculates precision for each class
   in a confusion matrix"""
import numpy as np


def precision(confusion):
    """return precision for each class"""
    return np.sum((confusion * np.identity(
                   confusion.shape[0]))/np.sum(confusion, axis=0), axis=0)
