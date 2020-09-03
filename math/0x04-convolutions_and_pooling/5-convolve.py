#!/usr/bin/env python3
"""performs convolution of images with channels"""
import numpy as np
from math import ceil, floor


def convolve(images, kernels, padding='same', stride=(1, 1)):
    """returns convolved matrix"""
    hk = kernels.shape[0]
    wk = kernels.shape[1]
    nc = kernels.shape[3]
    m = images.shape[0]
    if padding == "valid":
        pad0 = 0
        pad1 = 0
    elif padding == "same":
        pad0 = int((hk - 1) / 2)
        pad1 = int((wk - 1) / 2)
    else:
        pad0 = padding[0]
        pad1 = padding[1]
    out_h = int(floor(float(images.shape[1] + 2 * pad0  - hk) / float(stride[0]))) + 1
    out_w = int(floor(float(images.shape[2] + 2 * pad1 - wk) / float(stride[1]))) + 1
    convoluted = np.zeros((m, out_h, out_w, nc))
    img = np.pad(images, ((0,0), (pad0,pad0), (pad1,pad1), (0,0)), 'constant')
    for c in range(nc):
        for h in range(out_h):
            for w in range(out_w):
                matrix = img[:, h * stride[0] : h * stride[0] + hk, w * stride[1] : w * stride[1] + wk, :]
                v = np.sum(matrix * kernels[..., c], axis=1).sum(axis=1).sum(axis=1)
                convoluted[:, h, w, c] = v
    return(convoluted)
