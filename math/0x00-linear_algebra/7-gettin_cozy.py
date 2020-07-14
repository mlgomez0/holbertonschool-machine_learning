#!/usr/bin/env python3
"""This function concatenates two matrices"""


def cat_matrices2D(mat1, mat2, axis=0):
    """returns a new concatenated matrix"""
    concat = []
    matrix = []
    if axis == 0 and len(mat1[0]) == len(mat2[0]):
        for item in mat1:
            concat.append(item[:])
        for row in mat2:
            concat.append(row[:])
        return concat
    if axis == 1 and len(mat1) == len(mat2):
        for i in range(len(mat1)):
            rowx = mat1[i][:]
            for j in range(len(mat2[0])):
                rowx.append(mat2[i][j])
            matrix.append(rowx)
        return matrix
    return None
