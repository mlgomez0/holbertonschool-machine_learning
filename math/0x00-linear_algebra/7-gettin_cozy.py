#!/usr/bin/env python3
"""This function concatenates two matrices"""


def cat_matrices2D(mat1, mat2, axis=0):
    """returns a new concatenated matrix"""
    concat = mat1[:]
    #mat1c = mat1[:]
    matrix = []
    if axis == 0 and len(mat1[0]) == len(mat2[0]):
        for row in mat2:
            concat.append(row)
        return concat
    if axis == 1 and len(mat1) == len(mat2):
        for i in range(len(mat1)):
            rowx = mat1[i][:]
            for j in range(len(mat2[0])):
                rowx.append(mat2[i][j])
            matrix.append(rowx)
        return matrix          
        #return [[mat1c[i].append(mat2[i][j]) for j in range(len(mat2[0]))] for i in range(len(mat2))]
    return None
