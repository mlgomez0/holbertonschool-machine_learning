#!/usr/bin/env python3
"""performs forward propagation over a
   convolutional layer of a neural network"""
import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """the output of the convolutional layer"""
    hk = W.shape[0]
    wk = W.shape[1]
    nc = W.shape[3]
    m = A_prev.shape[0]
    hm = A_prev.shape[1]
    wm = A_prev.shape[2]
    st1 = stride[1]
    st0 = stride[0]
    if padding == "valid":
        pad0 = 0
        pad1 = 0
    else:
        pad0 = int(((hm - 1) * st0 + hk - hm) / 2) + 1
        pad1 = int(((wm - 1) * st1 + wk - wm) / 2) + 1
    out_h = int((hm + 2 * pad0 - hk) / st0) + 1
    out_w = int((wm + 2 * pad1 - wk) / st1) + 1
    convoluted = np.zeros((m, out_h, out_w, nc))
    img = np.pad(A_prev, ((0, 0), (pad0, pad0), (
                           pad1, pad1), (0, 0)), 'constant')
    for c in range(nc):
        for h in range(out_h):
            for w in range(out_w):
                ma = img[:, h * st0: h * st0 + hk, w * st1: w * st1 + wk, :]
                vm = np.sum(ma * W[..., c], axis=1).sum(axis=1)
                v = vm.sum(axis=1)
                convoluted[:, h, w, c] = v
    z = convoluted + b
    return(activation(z))
