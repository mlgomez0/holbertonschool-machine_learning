"""updates the weights of a neural network with 
   Dropout regularization using gradient descent"""
import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """weights of the network should be 
       updated in place"""