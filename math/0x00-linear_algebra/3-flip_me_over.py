#!/usr/bin/env python3
"""This function returns the transpose of a 2D matrix"""


def matrix_transpose(matrix):
    """matrix elements are of the same type/shape"""
    transpose = []
    for i in range(len(matrix[0])):
        row = []
        for item in matrix:
            row.append(item[i])
        transpose.append(row)
    return transpose
