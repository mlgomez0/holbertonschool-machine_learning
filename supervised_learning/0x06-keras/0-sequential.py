#!/usr/bin/env python3
import tensorflow.keras as K

def build_model(nx, layers, activations, lambtha, keep_prob):
    obj = K.Sequential()
    regularization = K.regularizers.l2(lambtha)
    for i in range(len(layers)):
        if i == 0:
            obj.add(K.layers.Dense(layers[i], activation=activations[i], kernel_regularizer=regularization, input_dim=nx))
        else:
            obj.add(K.layers.Dense(layers[i], activation=activations[i], kernel_regularizer=regularization))
        if i < len(layers) - 1:
            obj.add(K.layers.Dropout(keep_prob))
    return obj
