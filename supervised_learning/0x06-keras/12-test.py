#!/usr/bin/env python3


def test_model(network, data, labels, verbose=True):
    return network.evaluate(data, labels, verbose=verbose)
