#!/usr/bin/env python3
"""performs convolution of gray images using custom padding"""
import numpy as np
from math import ceil, floor


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """returns convolved matrix"""
    hk = kernel.shape[0]
    wk = kernel.shape[1]
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
    st1 = stride[1]
    st0 = stride[0]
    out_h = int(floor(float(
                images.shape[1] + 2 * pad0 - hk) / float(
                st0))) + 1
    out_w = int(floor(float(
                images.shape[2] + 2 * pad1 - wk) / float(
                st1))) + 1
    convoluted = np.zeros((m, out_h, out_w))
    img = np.pad(images, ((0, 0), (pad0, pad0), (pad1, pad1)), 'constant')
    for h in range(out_h):
        for w in range(out_w):
            matrix = img[:, h * st0: h * st0 + hk, w * st1: w * st1 + wk]
            v = np.sum(matrix * kernel, axis=1).sum(axis=1)
            convoluted[:, h, w] = v
    return(convoluted)
