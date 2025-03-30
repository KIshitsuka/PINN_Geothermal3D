"""
This module imprements the normalizer for temperature, pressure, logarithm of permeability
"""

from abc import ABC, abstractmethod

import tensorflow as tf



class Normalizer(ABC):
    """
    Basic class for normalizer

    """

    @abstractmethod
    def normalize(self, x):
        """
        Normalize target quantity

        Parameters
        ----------
        x:
            Target quantity to be normalized

        Returns
        -------
        normalied_x:
            Normalized target quantity
        """
        pass

    @abstractmethod
    def denormalize(self, x):
        """
        Denormalize normalized quantity

        Parameters
        ----------
        x:
            Normalized quantity

        Returns
        -------
        normalied_x:
            Normalized target quantity
        """
        pass


class MinMaxNormalizer(Normalizer):
    """
    Convert target quantity to be minimum of 0 and maximum of 1
    """

    def __init__(self, min_value, max_value):
        if max_value <= min_value:
            raise ValueError(
                f"max_value should be greater than min_value, but man_value(={max_value}) <= min_value(={min_value})"
            )

        self.min_value = min_value
        self.max_value = max_value

    def normalize(self, x):
        return (x - self.min_value) / (self.max_value - self.min_value)

    def denormalize(self, x):
        return x * (self.max_value - self.min_value) + self.min_value

