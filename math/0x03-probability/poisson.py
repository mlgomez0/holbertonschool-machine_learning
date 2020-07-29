#!/usr/bin/env python3
"""Poisson class"""


class Poisson:
    """this class allows poisson objects"""
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
                self.lambtha = sum(data) / len(data)

    def pmf(self, k):
        """calculates the PMF given a number of successes: k"""
        if k < 0:
            return 0
        if type(k) != int:
            k = int(k)
        factorial_k = 1
        if k != 0:
            for i in range(1, k + 1):
                factorial_k = factorial_k * i
        return ((2.7182818285**(-(self.lambtha)))*(
            self.lambtha**(k))) / factorial_k

    def cdf(self, k):
        """calculates the CDF given a number of successes: k"""
        if k < 0:
            return 0
        if type(k) != int:
            k = int(k)
        CDF = 0
        for i in range(0, k+1):
            CDF = CDF + self.pmf(i)
        return CDF
