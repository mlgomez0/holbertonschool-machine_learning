#!/usr/bin/env python3
"""Performs accuracy"""
import tensorflow as tf
create_layer = __import__('1-create_layer').create_layer

def calculate_accuracy(y, y_pred):
    """tensor with accuracy"""
    Ac = tf.losses.softmax_cross_entropy(y, y_pred)
    return Ac
