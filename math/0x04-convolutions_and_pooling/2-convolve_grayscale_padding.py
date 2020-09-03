#!/usr/bin/env python3
"""performs convolution of gray images with custom padding"""
import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """returns convolved images"""
    hk = kernel.shape[0]
    wk = kernel.shape[1]
    m = images.shape[0]
    out_h = images.shape[1] + 2 * padding[0] - hk + 1
    out_w = images.shape[2] + 2 * padding[1] - wk + 1
    convoluted = np.zeros((m, out_h, out_w))
    img = np.pad(images, (
                          (0, 0), (padding[0], padding[0]), (
                           padding[1], padding[1])), 'constant')
    for h in range(out_h):
        for w in range(out_w):
            matrix = img[:, h: h + hk, w: w + wk]
            v = np.sum(matrix * kernel, axis=1).sum(axis=1)
            convoluted[:, h, w] = v
    return(convoluted)
