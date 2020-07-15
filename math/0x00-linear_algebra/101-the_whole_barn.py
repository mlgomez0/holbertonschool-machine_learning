#!/usr/bin/env python3
"""adds two matrices"""
import numpy as np


def matrix_shape(matrix):
    """matrix elements are of the same type/shape"""
    dimensions = []
    len_matrix = len(matrix)
    dimensions.append(len_matrix)
    while type(matrix[0]) == list:
        matrix = matrix[0]
        len_matrix = len(matrix)
        dimensions.append(len_matrix)
    return dimensions

def add_matrices(mat1, mat2):
    """returns new matrix"""
    shape1 = matrix_shape(mat1)
    shape2 = matrix_shape(mat2)
    for i in range(len(shape1)):
        if shape1[i] != shape2[i] or len(shape1) != len(shape2):
            return None

    return (np.add(mat1, mat2))
