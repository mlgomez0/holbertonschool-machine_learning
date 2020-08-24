#!/usr/bin/env python3
"""calculates specificity for each class
   in a confusion matrix"""
import numpy as np


def specificity(confusion):
    """return specificity for each class"""
    id_ma = np.identity(confusion.shape[0])
    zero_diagonal = np.where(confusion * id_ma == 0, confusion, 0)
    false_positives = np.sum(zero_diagonal, axis=0)
    true_negatives = []
    for i in range(confusion.shape[0]):
        if i == 0:
            true_negatives.append(np.sum(confusion[1:, 1:]))
        elif i == confusion.shape[0] - 1:
            true_negatives.append(np.sum(confusion[0:i, 0:i]))
        else:
            true_negatives.append(np.sum(confusion[0:i, 0:i]) + np.sum(
                                  confusion[0:i, i + 1:]) + np.sum(
                                  confusion[i + 1:, i + 1:]) + np.sum(
                                  confusion[i + 1:, 0:i]))
    return true_negatives / (true_negatives + false_positives)
