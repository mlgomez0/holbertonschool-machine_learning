#!/usr/bin/env python3
"""returns moving averages with bias correction"""
import numpy as np

def moving_average(data, beta):
    vp = 0
    weighted = []
    for i in range(len(data)):
        v = (vp * beta + (1 - beta) * data[i])
        weighted.append(v / (1 - beta**(i + 1)))
        vp = v
    return weighted
