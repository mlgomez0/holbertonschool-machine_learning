#!/usr/bin/env python3
"""Performs accuracy"""
import tensorflow as tf

def calculate_accuracy(y, y_pred):
    """tensor with accuracy"""
    equal = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(equal, "float"))
    return accuracy
