"""Twoport network module."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import skrf as rf
from pydantic import BaseModel, ConfigDict, NonNegativeInt

from twpasolver.abcd_matrices import ABCDArray
from twpasolver.file_utils import read_file, save_to_file
from twpasolver.mathutils import a2s, s2a
from twpasolver.typing import Impedance


class TwoPortCell:
    """Class representing a two-port RF cell."""

    def __init__(
        self, freqs: np.ndarray, abcd: np.ndarray | ABCDArray, Z0: Impedance = 50
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

    @classmethod
    def from_file(cls, filename: str, writer="hdf5"):
        """Load model from file."""
        model_dict = read_file(filename, writer=writer)
        return cls(model_dict["freqs"], model_dict["abcd"], model_dict["Z0"])

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

    def to_network(self):
        """Convert to scikit-rf Network."""
        f = rf.Frequency.from_f(self.freqs * 1e-9, "ghz")
        return rf.Network(frequency=f, a=np.asarray(self.abcd))

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

    def as_dict(self):
        """Return cell contents as dictionary."""
        return {"freqs": self.freqs, "abcd": np.asarray(self.abcd), "Z0": self.Z0}

    def dump_to_file(self, filename: str, writer="hdf5"):
        """Dump cell to file."""
        save_to_file(filename, self.as_dict(), writer=writer)


class TwoPortModel(BaseModel, ABC):
    """Base class for models of two-port networks."""

    model_config = ConfigDict(
        validate_assignment=True,
        revalidate_instances="always",
    )
    name: str | None = None
    Z0: Impedance = 50.0
    N: NonNegativeInt = 1

    @classmethod
    def from_file(cls, filename: str):
        """Load model from file."""
        model_dict = read_file(filename, writer="json")
        return cls(**model_dict)

    def update(self, **kwargs) -> None:
        """Update multiple attributes of the model."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise RuntimeError(f"The cell model does not have the {key} attribute.")

    @abstractmethod
    def single_abcd(self, freqs: np.ndarray) -> ABCDArray:
        """Compute the abcd matrix of a single iteration of the model."""

    def get_abcd(self, freqs: np.ndarray) -> ABCDArray:
        """Compute the abcd matrix of the model."""
        if self.N == 1:
            return self.single_abcd(freqs)
        return self.single_abcd(freqs) ** self.N

    def get_cell(self, freqs: np.ndarray) -> TwoPortCell:
        """Return the two-port cell of the model."""
        return TwoPortCell(freqs, self.get_abcd(freqs), Z0=self.Z0)

    def get_network(self, freqs: np.ndarray) -> rf.Network:
        """Return the two-port cell of the model as a scikit-rf Network."""
        return self.get_cell(freqs).to_network()

    def dump_to_file(self, filename: str):
        """Dump model to file."""
        model_dict = self.model_dump()
        save_to_file(filename, model_dict, writer="json")
