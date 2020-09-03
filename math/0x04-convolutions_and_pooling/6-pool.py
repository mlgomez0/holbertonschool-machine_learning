#!/usr/bin/env python3
"""performs convolution images using pooling"""
import numpy as np
from math import ceil, floor


def pool(images, kernel_shape, stride, mode='max'):
    """returns convolved matrix"""
    hk = kernel_shape[0]
    wk = kernel_shape[1]
    image_h = images.shape[1]
    image_w = images.shape[2]
    nc = images.shape[3]
    m = images.shape[0]
    out_h = int(floor(float(image_h - hk) / float(stride[0]))) + 1
    out_w = int(floor(float(image_w - wk) / float(stride[1]))) + 1
    pooled = np.zeros((m, out_h, out_w, nc))
    img = images.copy()
    for h in range(out_h):
        for w in range(out_w):
            matrix = img[:, h * stride[0] : h * stride[0] + hk, w * stride[1] : w * stride[1] + wk,:]
            if mode == "max":
                v = np.max(matrix, axis=(1,2))
            else:
                v = np.average(matrix, axis=(1,2))
            pooled[:, h, w,:] = v
    return(pooled)
