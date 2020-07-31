#!/usr/bin/env python3
"""class Neuron"""
import numpy as np


class Neuron:
    """defines single neuron for Binary Classification"""
    def __init__(self, nx):
        if type(nx) != int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        self.W = np.random.randn(1, nx)
        self.b = 0
        self.A = 0
