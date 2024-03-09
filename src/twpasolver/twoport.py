"""Twoport network module."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from twpasolver.abcd_matrices import ABCDArray
from twpasolver.mathutils import a2s, s2a


class TwoPortCell:
    """Class representing a two-port RF cell."""

    def __init__(
        self, freqs: np.ndarray, abcd: np.ndarray | ABCDArray, Z0: complex | float = 50
    ):
        """
        Initialize the TwoPortCell instance.

        Parameters:
        - freqs (numpy.ndarray): Frequencies of the network.
        - mat (numpy.ndarray): Input array of 2x2 matrices.
        - Z0 (float or int): Line impedance.
        """
        if not isinstance(abcd, ABCDArray):
            abcd = ABCDArray(abcd)
        if freqs.shape[0] != abcd.shape[0]:
            raise ValueError("Frequencies and abcd matrices must have same length.")
        self.abcd = abcd
        self.freqs = freqs
        self.Z0 = Z0

    @classmethod
    def from_s_par(cls, freqs: np.ndarray, s_mat: np.ndarray, Z0: float | int = 50):
        """Instantiate from array of S-parameters."""
        abcd_mat = s2a(s_mat, Z0)
        return cls(freqs, abcd_mat, Z0=Z0)

    @property
    def freqs(self) -> np.ndarray:
        """Frequencies array getter."""
        return self._freqs

    @freqs.setter
    def freqs(self, freqs: np.ndarray):
        if freqs.ndim != 1:
            raise ValueError("Frequencies must be 1-D array")
        if min(freqs) < 0:
            raise ValueError("Frequencies must be positive numbers")
        if freqs.shape[0] != self.abcd.shape[0]:
            raise ValueError("Frequencies and abcd matrices must have same length.")
        self._freqs = np.asarray(freqs)

    @property
    def Z0(self) -> complex | float:
        """Line impedance getter."""
        return self._Z0

    @Z0.setter
    def Z0(self, value: complex | float):
        if np.real(value) <= 0:
            raise ValueError("Resistive component of line impedance must be positive.")
        self._Z0 = value

    def get_s_par(self):
        """Return S-parameter matrix."""
        return a2s(np.asarray(self.abcd), self.Z0)

    def __repr__(self):
        """Return a string representation of the TwoPortCell."""
        return f"{self.__class__.__name__}(freqs={self.freqs},\nabcd={self.abcd},\nZ0={self.Z0})"

    def __getitem__(self, idxs):
        """Get slice of TwoPortCell."""
        if not isinstance(idxs, slice):
            raise ValueError("Only slicing of TwoPortCell is allowed")
        return self.__class__(self.freqs[idxs], self.abcd[idxs], self.Z0)


@dataclass
class TwoPortModel(ABC):
    """Base class for models of two-port networks."""

    def update(self, **kwargs) -> None:
        """Update multiple attributes of the model."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise RuntimeError(f"The cell model does not have the {key} attribute.")

    @abstractmethod
    def get_abcd(self, freqs: np.ndarray) -> TwoPortCell:
        """Compute the abcd matrix of the model."""
