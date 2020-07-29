#!/usr/bin/env python3
"""Exponential class"""


class Exponential:
    """this class allows exponential distribution"""
    def __init__(self, data=None, lambtha=1.):
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            else:
                self.lambtha = float(lambtha)
        else:
            if type(data) != list:
                raise TypeError("data must be a list")
            elif len(data) < 2:
                raise ValueError("data must contain multiple values")
            else:
                self.lambtha = 1 / (sum(data) / len(data))

    def pdf(self, x):
        """calculates the PDF fo a given time period: x"""
        if x < 0:
            return 0
        return self.lambtha * 2.7182818285**(-x * self.lambtha)

    def cdf(self, x):
        """calculates the CDF given a time period: x"""
        if x < 0:
            return 0
        return 1 - 2.7182818285**(-x * self.lambtha)
