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
            self.__weights[w] = np.random.randn(i, layer_size) * np.sqrt(2/layer_size)
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

    def forward_prop(self, X):
        self.__cache["A0"] = X
        for i in range(1, self.__L + 1):
            w = "W" + str(i)
            b = "b" + str(i)
            a = "A" + str(i - 1)
            Z = np.matmul(self.__weights[w], self.__cache[a]) + self.__weights[b]
            a_new = "A" + str(i)
            self.__cache[a_new] = 1 / (1 + np.exp(-Z))
        l = "A" + str(self.__L)
        return (self.__cache[l], self.__cache)

    def cost(self, Y, A):
        cost_array = (np.log(A) * Y) + ((1 - Y) * np.log(1.0000001 - A))
        cost = -np.sum(cost_array) / len(A[0])
        return cost

    def evaluate(self, X, Y):
        A, _ = self.forward_prop(X)
        cost_r = self.cost(Y, A)
        return (np.where(A >= 0.5, 1, 0), cost_r)

    def gradient_descent(self, Y, cache, alpha=0.05):
        da = -(Y/cache["A" + str(self.__L)]) + ((1 - Y)/(1 - cache["A" + str(self.__L)]))
        for i in range(self.__L, 0, -1):
            Al = "A" + str(i)
            dz = da * (cache[Al] * (1 - cache[Al]))
            Al_ = "A" + str(i - 1)
            dw = (1 / len(Y[0])) * np.matmul(dz, cache[Al_].T)
            db = (1 / len(Y[0])) * np.sum(dz, axis=1, keepdims=True)
            self.__weights["W" + str(i)] = self.__weights["W" + str(i)] - alpha * dw
            self.__weights["b" + str(i)] = self.__weights["b" + str(i)] - alpha * db
            da = np.matmul(self.__weights["W" + str(i)].T, dz)
