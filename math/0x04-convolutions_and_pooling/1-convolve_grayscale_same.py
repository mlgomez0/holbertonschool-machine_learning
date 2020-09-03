#!/usr/bin/env python3
"""performs convolution of gray images using padding"""
import numpy as np


def convolve_grayscale_same(images, kernel):
    """returns convolved imges"""
    out_h = images.shape[1]
    out_w = images.shape[2]
    hk = kernel.shape[0]
    wk = kernel.shape[1]
    pad_h = int(hk / 2)
    pad_w = int(wk / 2)
    m = images.shape[0]
    convoluted = np.zeros((m, out_h, out_w))
    img = np.pad(images, ((0, 0), (pad_h, pad_h), (pad_w, pad_w)), 'constant')
    for h in range(out_h):
        for w in range(out_w):
            matrix = img[:, h: h + hk, w: w + wk]
            v = np.sum(matrix * kernel, axis=1).sum(axis=1)
            convoluted[:, h, w] = v
    return(convoluted)
