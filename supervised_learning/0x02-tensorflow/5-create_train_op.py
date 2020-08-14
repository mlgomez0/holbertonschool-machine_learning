#!/usr/bin/env python3
"""creates training operation"""
import tensorflow as tf


def create_train_op(loss, alpha):
    """operations that trains a model"""
    optimizer = tf.train.GradientDescentOptimizer(alpha)
    train = optimizer.minimize(loss)
    return train
