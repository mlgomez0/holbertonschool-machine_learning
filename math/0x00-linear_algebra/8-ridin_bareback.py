#!/usr/bin/env python3
"""This function multiply two matrices"""


def mat_mul(mat1, mat2):
    """returns new matrix or None"""
    if len(mat1[0]) != len(mat2):
        return None
    matrix = [[0 for i in range(len(mat2[0]))] for y in range(len(mat1))]
    for i in range(len(mat1)):
        for j in range(len(mat2[0])):
            n = 0
            sum_ele = 0
            while n < len(mat2):
                sum_ele += mat1[i][n] * mat2[n][j]
                n += 1
            matrix[i][j] = sum_ele
    return matrix
