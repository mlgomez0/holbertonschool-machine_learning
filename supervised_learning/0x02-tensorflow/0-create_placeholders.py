#!/usr/bin/env python3
"""Creates two placeholders"""
import tensorflow as tf


def create_placeholders(nx, classes):
    """returns placeholders for NN"""
    x = tf.placeholder(tf.float32, shape=(None, nx), name="x")
    y = tf.placeholder(tf.float32, shape=(None, classes), name="y")
    return (x, y)
