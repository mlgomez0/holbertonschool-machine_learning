#!/usr/bin/env python3
"""calculates the cost of a neural network
   with L2 regularization:"""
import numpy as np
import tensorflow as tf


def l2_reg_cost(cost):
    """Returns: a tensor containing the cost
       of the network accounting for L2 regularization"""
    regularization_losses = tf.losses.get_regularization_losses()
    return cost + regularization_losses
