#!/usr/bin/env python3
"""Performs forwards propagation"""
import tensorflow as tf
create_layer = __import__('1-create_layer').create_layer

def forward_prop(x, layer_sizes=[], activations=[]):
    """returns the prediction of the NN"""
    A = x
    for i in range(len(layer_sizes)):
        A = create_layer(A, layer_sizes[i], activations[i])
    return A
