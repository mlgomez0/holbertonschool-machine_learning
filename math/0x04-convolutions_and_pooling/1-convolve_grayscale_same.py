#!/usr/bin/env python3
import numpy as np

def convolve_grayscale_same(images, kernel):
    images = np.pad(images, 1, mode="constant")
    m = images.shape[0]
    hm = images.shape[1]
    wm = images.shape[2]
    hk = kernel.shape[0]
    wk = kernel.shape[1]
    p = (hk - 1) / 2
    ch = int(hm - + 2*p + hk + 1)
    print(ch)
    convoluted = np.zeros((m, wm, wm))
    for h in range(wm):
        for w in range(wm):
            matrix = images[:, h : h + hk, w : w + wk]
            v = np.sum(matrix * kernel, axis=1).sum(axis=1)
            convoluted[:, h, w] = v
    return(convoluted)
