#!/usr/bin/env python3
"""class DeepNeuralNetwork with multiple hidden layer"""
import numpy as np


class DeepNeuralNetwork:
    """defines deep Neural Network for Binary Classification"""
    def __init__(self, nx, layers):
        if type(nx) != int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(layers) != list:
            raise TypeError("layers must be a list of positive integers")
        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}
        num_layer = 1
        layer_size = nx
        for i in layers:
            if type(i) != int or i < 0:
                raise TypeError("layers must be a list of positive integers")
            w = "W" + str(num_layer)
            b = "b" + str(num_layer)
            self.__weights[w] = np.random.randn(
                i, layer_size) * np.sqrt(2/layer_size)
            self.__weights[b] = np.zeros((i, 1))
            num_layer += 1
            layer_size = i

    @property
    def L(self):
        return self.__L

    @property
    def cache(self):
        return self.__cache

    @property
    def weights(self):
        return self.__weights
