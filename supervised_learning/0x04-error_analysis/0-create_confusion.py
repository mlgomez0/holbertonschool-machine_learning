#!/usr/bin/env python3
"""function that creates a confusion matrix"""
import numpy as np


def create_confusion_matrix(labels, logits):
    """return confusion matrix"""
    return np.matmul(labels.T, logits)
