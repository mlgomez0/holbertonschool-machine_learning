#!/usr/bin/env python3
"""Creates a layer"""
import tensorflow as tf


def create_layer(prev, n, activation):
    """returns tensor ourput layer"""
    w = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    layer = tf.layers.Dense(units=n, activation=activation,
                            kernel_initializer=w, name="layer")
    y = layer(prev)
    return (y)
