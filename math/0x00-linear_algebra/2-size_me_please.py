#!/usr/bin/env python3
"""This function calculates the size of a given matrix"""


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
