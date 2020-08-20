#!/usr/bin/env python3
"""matrix normalization"""


def normalize(X, m, s):
    """return normalized X"""
    return (X - m) / s
