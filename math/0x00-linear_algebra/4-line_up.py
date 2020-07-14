#!/usr/bin/env python3
"""This function adds two arrays"""


def add_arrays(arr1, arr2):
    """returns a list or none"""
    addition = []
    if len(arr1) != len(arr2):
        return None
    for i in range(len(arr1)):
        addition.append(arr1[i] + arr2[i])
    return addition
