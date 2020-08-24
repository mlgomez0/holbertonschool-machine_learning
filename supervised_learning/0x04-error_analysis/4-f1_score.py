#!/usr/bin/env python3
"""function that calculates F1 score os a confusion matrix"""
import numpy as np
sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision


def f1_score(confusion):
    """return fi score"""
    return 2 * (sensitivity(confusion) * precision(confusion)) / (
                sensitivity(confusion) + precision(confusion))
