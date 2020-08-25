#!/usr/bin/env python3
"""updates the weights and biases of a neural network
   using gradient descent with L2 regularization:"""
import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """weights and biases of the network
       should be updated in place"""
    weights_copy = weights.copy()
    dz = cache["A" + str(L)] - Y
    for i in range(L, 0, -1):
        A = cache["A" + str(i - 1)]
        dw = (1 / len(Y[0])) * np.matmul(dz, A.T)
        db = (1 / len(Y[0])) * np.sum(dz, axis=1, keepdims=True)
        w = "W" + str(i)
        b = "b" + str(i)
        weights[w] = weights[w] - (
            alpha * lambtha) / len(Y[0]) * weights[w] - alpha * dw
        weights[b] = weights[b] - (
            alpha * lambtha) / len(Y[0]) * weights[b] - alpha * db
        dz = np.matmul(weights_copy["W" + str(i)].T, dz) * (1 - A * A)
