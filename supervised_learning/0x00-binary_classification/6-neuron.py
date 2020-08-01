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
        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        return self.__W

    @property
    def b(self):
        return self.__b

    @property
    def A(self):
        return self.__A

    def forward_prop(self, X):
        Z = np.matmul(self.__W, X) + self.__b
        self.__A = 1 / (1 + np.exp(-Z))
        return self.__A

    def cost(self, Y, A):
        cost_array = np.multiply(np.log(A), Y) + np.multiply((1 - Y), np.log(1.0000001 - A))
        cost = -np.sum(cost_array) / len(A[0])
        return cost

    def evaluate(self, X, Y):
        self.forward_prop(X)
        cost_r = self.cost(Y, self.__A)
        return (np.where(self.__A >= 0.5, 1, 0), cost_r)

    def gradient_descent(self, X, Y, A, alpha=0.05):
        dz = A - Y
        dw = (1 / len(Y[0])) * np.matmul(dz, X.T)
        db = (1 / len(Y[0])) * np.sum(dz)
        self.__W = self.__W - alpha * dw
        self.__b = self.__b - alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05):
        if type(iterations) != int:
            raise TypeError("iterations must be an integer")
        if iterations < 0:
            raise ValueError("iterations must be a positive integer")
        if type(alpha) != float:
            raise TypeError("alpha must be a float")
        if alpha < 0:
            raise ValueError("alpha must be positive")
        self.forward_prop(X)
        for i in range(iterations):
            self.gradient_descent(X, Y, self.__A, alpha)
            self.forward_prop(X)
        return self.evaluate(X, Y)
