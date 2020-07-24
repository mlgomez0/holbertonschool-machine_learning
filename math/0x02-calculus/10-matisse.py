#!/usr/bin/env python3

"""calculates the derivative of a polynomial"""

def poly_derivative(poly):
    """returns a list with the derivative coefficients"""
    deriv = []
    i = 1
    if type(poly) != list or len(poly) == 0 or poly[0] == 0 or poly[len(poly) - 1] == 0:
        return None
    if len(poly) == 1:
        return [0]

    while i < len(poly):
        deriv.append(i * poly[i])
        i += 1
    return deriv