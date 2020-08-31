#!/usr/bin/env python3
import tensorflow.keras as K

def optimize_model(network, alpha, beta1, beta2):
    opt = K.optimizers.Adam(lr=alpha, beta_1=beta1, beta_2=beta2)
    network.compile(loss='categorical_crossentropy', optimizer=opt, metrics=[K.metrics.Accuracy()])
    return None
