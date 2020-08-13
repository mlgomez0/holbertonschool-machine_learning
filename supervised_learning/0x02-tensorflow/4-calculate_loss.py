#!/usr/bin/env python3
"""calculates loss"""
import tensorflow as tf

def calculate_loss(y, y_pred):
    """tensor with loss"""
    loss = tf.losses.softmax_cross_entropy(y, y_pred)
    return loss
