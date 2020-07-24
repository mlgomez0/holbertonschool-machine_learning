#!/usr/bin/env python3
"""function calculates the summation of i**2 starting from 1"""


def summation_i_squared(n):
    """returns the summation"""
    if type(n) != int:
        return None
    if n == 0:
        return 1
    if n < 0:
        n = n * -1
    if n == 1:
        return 1
    summation = n**2
    return summation + summation_i_squared(n - 1)
