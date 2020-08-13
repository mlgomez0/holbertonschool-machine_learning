#!/usr/bin/env python3
"""Creates a layer"""
import tensorflow as tf


def create_layer(prev, n, activation):
    """returns tensor ourput layer"""
    layer = tf.layers.Dense(units=n, activation=activation, kernel_initializer=tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG"), name="layer")
    y = layer(prev)
    return (y)
