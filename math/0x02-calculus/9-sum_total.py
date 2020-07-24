#!/usr/bin/env python3
"""function calculates the summation of i**2 starting from 1"""


def summation_i_squared(n):
    """returns the summation"""
    if type(n) != int:
        return None
    if n < 1:
        return None
    return int((n * (n + 1) * (2 * n + 1)) / 6)
