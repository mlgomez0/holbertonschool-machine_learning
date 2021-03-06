#!/usr/bin/env python3
"""calculates the cost of a neural network
   with L2 regularization"""
import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """Returns: the cost of the network
       accounting for L2 regularization"""
    sum_norms = 0
    for i in range(1, L + 1):
        w = "W" + str(i)
        sum_norms = sum_norms + (np.linalg.norm(weights[w]))

    return cost + (sum_norms * (lambtha / (2 * m)))
