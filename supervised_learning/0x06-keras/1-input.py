#!/usr/bin/env python3
import tensorflow.keras as K

def build_model(nx, layers, activations, lambtha, keep_prob):
    inputs = K.Input(shape=(nx,))
    regularization = K.regularizers.l2(lambtha)
    for i in range(0, len(layers)):
        if i == 0:
            x = K.layers.Dense(layers[i], activation=activations[i], kernel_regularizer=regularization)(inputs)
            x = K.layers.Dropout(keep_prob)(x)
        elif (i < len(layers) - 1):
            x = K.layers.Dense(layers[i], activation=activations[i], kernel_regularizer=regularization)(x)
            x = K.layers.Dropout(keep_prob)(x)
        else:
            outputs = K.layers.Dense(layers[i], activation=activations[i], kernel_regularizer=regularization)(x)
    return K.Model(inputs=inputs, outputs=outputs)
