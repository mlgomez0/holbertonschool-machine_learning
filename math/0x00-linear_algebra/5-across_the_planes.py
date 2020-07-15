#!/usr/bin/env python3
"""This function adds two matrices"""


def add_matrices2D(mat1, mat2):
    """returns a matrix or none"""
    addition = []
    if len(mat1) != len(mat2) or len(mat1[0]) != len(mat2[0]):
        return None
    m = [[mat1[i][j] + mat2[i][j] for j in range(len(
        mat2[0]))] for i in range(len(mat2))]
    return m
