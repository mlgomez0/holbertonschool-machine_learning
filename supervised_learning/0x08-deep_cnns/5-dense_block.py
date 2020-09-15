#!/usr/bin/env python3
"""builds the inception network as described in Going Deeper
   with Convolutions (2014)"""
import tensorflow.keras as K


def dense_block(X, nb_filters, growth_rate, layers):
    """return"""
    filter_num = nb_filters
    con_X = X
    for _ in range(layers):
        BN1 = K.layers.BatchNormalization(axis=3)(con_X)
        Relu1 = K.layers.Activation("relu")(BN1)
        conv1 = K.layers.Conv2D(filters= 4 * growth_rate,
                                kernel_size=(1, 1),
                                padding="same",
                                kernel_initializer="he_normal")(Relu1)
        BN1 = K.layers.BatchNormalization(axis=3)(conv1)
        Relu2 = K.layers.Activation("relu")(BN1)
        conv2 = K.layers.Conv2D(filters = growth_rate,
                                kernel_size=(3, 3),
                                padding="same",
                                kernel_initializer="he_normal")(Relu2)
        con_X = K.layers.Concatenate(axis=3)([con_X, conv2])
        filter_num += growth_rate
    return con_X, filter_num
