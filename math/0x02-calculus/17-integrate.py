#!/usr/bin/env python3
"""calculates integral of a polinomial"""


def poly_integral(poly, C=0):
    """returns a list with coefficients"""
    inte = []
    i = 0
    if type(poly) != list or len(poly) == 0 or type(C) != int:
        return None
    if len(poly) == 1:
        return poly
    inte.append(C)
    while i < len(poly):
        if poly[i] % (i + 1) == 0:
            inte.append(int(poly[i]/(i + 1)))
        else:
            inte.append(poly[i]/(i + 1))
        i += 1
    return inte
