#!/usr/bin/env python3
"""returns moving averages"""
import numpy as np

def moving_average(data, beta):
    vp = 0
    weighted = []
    for i in range(1, len(data)):
        v = (vp * beta + (1 - beta) * data[i])
        weighted.append(v)
        vp = v
    return weighted
