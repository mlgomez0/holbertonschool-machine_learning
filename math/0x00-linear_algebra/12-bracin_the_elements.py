#!/usr/bin/env python3
"""element-wise sum, difference, product, and quotient for matrices"""


def np_elementwise(mat1, mat2):
    """returns tupla element-wise sum, difference, product, and quotient"""
    return (mat1 + mat2, mat1-mat2, mat1 * mat2, mat1/mat2)
