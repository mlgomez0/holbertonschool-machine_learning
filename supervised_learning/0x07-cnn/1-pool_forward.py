#!/usr/bin/env python3
"""performs forward propagation over a 
   pooling layer of a neural network:"""
import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """returns output of the pooling layer"""
    hk = kernel_shape[0]
    wk = kernel_shape[1]
    nc = A_prev.shape[3]
    m = A_prev.shape[0]
    hm = A_prev.shape[1]
    wm = A_prev.shape[2]
    st1 = stride[1]
    st0 = stride[0]
    out_h = int((hm - hk) / st0) + 1
    out_w = int((wm - wk) / st1) + 1
    pooled = np.zeros((m, out_h, out_w, nc))
    img = A_prev.copy()
    for h in range(out_h):
        for w in range(out_w):
            matrix = img[:, h * st0: h * st0 + hk, w * st1: w * st1 + wk, :]
            if mode == "max":
                v = np.max(matrix, axis=(1, 2))
            else:
                v = np.average(matrix, axis=(1, 2))
            pooled[:, h, w, :] = v
    return(pooled)
