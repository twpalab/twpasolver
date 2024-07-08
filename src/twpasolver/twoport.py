"""
Two-port network module.

This module provides classes and methods for representing and manipulating two-port RF (radio frequency)
networks. It includes functionality for handling frequency-dependent network parameters, converting
between different representations (such as ABCD matrices and S-parameters), and performing common
operations on two-port networks.
This module can be used to define and work with two-port network models in RF electronics. It is
designed to support tasks such as network analysis, simulation, and design.

Examples
--------
Create a TwoPortCell from ABCD parameters:

.. code-block:: python

    freqs = np.array([1e9, 2e9, 3e9])
    abcd = np.array([
        [[1, 2], [3, 4]],
        [[5, 6], [7, 8]],
        [[9, 10], [11, 12]]
    ])
    cell = TwoPortCell(freqs, abcd)

Convert to S-parameters:

.. code-block:: python

    s_params = cell.s

Interpolate to new frequencies:

.. code-block:: python

    new_freqs = np.array([1.5e9, 2.5e9])
    interpolated_cell = cell.interpolate(new_freqs)

Save and load from file:

.. code-block:: python

    cell.dump_to_file('cell_data.hdf5')
    loaded_cell = TwoPortCell.from_file('cell_data.hdf5')
"""

from __future__ import annotations

import itertools
from abc import ABC, abstractmethod

import numpy as np
import skrf as rf
from pydantic import ConfigDict, Field, NonNegativeInt

from twpasolver.basemodel import BaseModel
from twpasolver.file_utils import read_file, save_to_file
from twpasolver.logging import log
from twpasolver.mathutils import a2s, s2a
from twpasolver.matrices_arrays import ABCDArray, SMatrixArray
from twpasolver.typing import Impedance, validate_impedance


class TwoPortCell:
    """Class representing a two-port RF cell."""

    def __init__(
        self,
        freqs: np.ndarray,
        abcd: np.ndarray | ABCDArray,
        Z0: Impedance = 50,
    ):
        """
        Initialize the TwoPortCell instance.

        Args:
            freqs (numpy.ndarray): Frequencies of the network.
            abcd (numpy.ndarray | ABCDArray): Input array of 2x2 matrices.
            Z0 (Impedance): Reference line impedance.
        """
        if not isinstance(abcd, ABCDArray):
            abcd = ABCDArray(abcd)
        if freqs.shape[0] != abcd.shape[0]:
            raise ValueError("Frequencies and abcd matrices must have same length.")
        self.abcd = abcd
        self.freqs = freqs
        self.Z0 = Z0

    @classmethod
    def from_s(
        cls, freqs: np.ndarray, s_mat: np.ndarray, Z0: float | int = 50
    ) -> TwoPortCell:
        """
        Instantiate from array of S-parameters.

        Args:
            freqs (numpy.ndarray): Frequencies of the network.
            s_mat (numpy.ndarray): S-parameter matrix.
            Z0 (float | int): Reference line impedance.

        Returns:
            TwoPortCell: Instance of TwoPortCell.
        """
        abcd_mat = s2a(s_mat, Z0)
        return cls(freqs, abcd_mat, Z0=Z0)

    @classmethod
    def from_file(cls, filename: str, writer="hdf5") -> TwoPortCell:
        """
        Load model from file.

        Args:
            filename (str): Name of the file.
            writer (str): File format.

        Returns:
            TwoPortCell: Instance of TwoPortCell.
        """
        model_dict = read_file(filename, writer=writer)
        return cls(model_dict["freqs"], model_dict["abcd"], model_dict["Z0"])

    @property
    def freqs(self) -> np.ndarray:
        """
        Get the frequencies array.

        Returns:
            numpy.ndarray: Frequencies array.
        """
        return self._freqs

    @freqs.setter
    def freqs(self, freqs: np.ndarray) -> None:
        """
        Set the frequencies.

        Args:
            freqs (numpy.ndarray): Frequencies array.
        """
        if freqs.ndim != 1:
            raise ValueError("Frequencies must be 1-D array")
        if min(freqs) < 0:
            raise ValueError("Frequencies must be positive numbers")
        if freqs.shape[0] != self.abcd.shape[0]:
            raise ValueError("Frequencies and abcd matrices must have same length.")
        self._freqs = np.asarray(freqs)

    @property
    def Z0(self) -> Impedance:
        """
        Get the reference line impedance.

        Returns:
            Impedance: Reference line impedance.
        """
        return self._Z0

    @Z0.setter
    def Z0(self, value: Impedance) -> None:
        """
        Set the reference line impedance.

        Args:
            value (Impedance): Reference line impedance.
        """
        validate_impedance(value)
        self._Z0 = value

    def to_network(self) -> rf.Network:
        """
        Convert to scikit-rf Network.

        Returns:
            rf.Network: scikit-rf Network.
        """
        f = rf.Frequency.from_f(self.freqs * 1e-9, "ghz")
        return rf.Network(frequency=f, a=np.asarray(self.abcd), z0=self.Z0)

    @property
    def s(self) -> SMatrixArray:
        """
        Get the S-parameter matrix.

        Returns:
            SMatrixArray: S-parameter matrix.
        """
        return SMatrixArray(a2s(np.asarray(self.abcd), self.Z0))

    def __repr__(self) -> str:
        """
        Return a string representation of the TwoPortCell.

        Returns:
            str: String representation of the TwoPortCell.
        """
        return f"{self.__class__.__name__}(freqs={self.freqs},\nabcd={self.abcd},\nZ0={self.Z0})"

    def __getitem__(self, idxs: slice) -> TwoPortCell:
        """
        Get slice of TwoPortCell.

        Args:
            idxs (slice): Slice indices.

        Returns:
            TwoPortCell: Sliced instance of TwoPortCell.
        """
        if not isinstance(idxs, slice):
            raise ValueError("Only slicing of TwoPortCell is allowed")
        return self.__class__(self.freqs[idxs], self.abcd[idxs], self.Z0)

    def as_dict(self) -> dict:
        """
        Return cell contents as dictionary.

        Returns:
            dict: Dictionary containing cell contents.
        """
        return {"freqs": self.freqs, "abcd": np.asarray(self.abcd), "Z0": self.Z0}

    def dump_to_file(self, filename: str, writer="hdf5") -> None:
        """
        Dump cell to file.

        Args:
            filename (str): Name of the file.
            writer (str): File format.
        """
        save_to_file(filename, self.as_dict(), writer=writer)

    def interpolate(self, freqs: np.ndarray, polar: bool = True) -> TwoPortCell:
        """
        Return abcd matrix of internal cell interpolating the given frequencies.

        Args:
            freqs (numpy.ndarray): Frequencies array to interpolate.
            polar (bool): Interpolate magnitude and phase instead of real and imaginary part.

        Returns:
            TwoPortCell: Interpolated instance of TwoPortCell.
        """
        if np.array_equal(freqs, self.freqs):
            return self
        if freqs[0] < self.freqs[0] or freqs[-1] > self.freqs[-1]:
            log.warning("Interpolation out of predefined range might be imprecise.")

        abcd_interp = []
        for i, j in itertools.product(range(2), repeat=2):
            if polar:
                interp_mag = np.interp(freqs, self.freqs, np.abs(self.abcd[:, i, j]))
                interp_phase = np.interp(
                    freqs, self.freqs, np.unwrap(np.angle(self.abcd[:, i, j]))
                )
                abcd_interp.append(interp_mag * np.exp(1j * interp_phase))
            else:
                abcd_interp.append(np.interp(freqs, self.freqs, self.abcd[:, i, j]))
        return self.__class__(freqs, np.array(abcd_interp), self.Z0)


class TwoPortModel(BaseModel, ABC):
    """Base class for models of two-port networks."""

    model_config = ConfigDict(
        validate_assignment=True, revalidate_instances="always", protected_namespaces=()
    )
    Z0_ref: Impedance = Field(
        default=50.0, description="Reference line impedance of the two-port component."
    )
    N: NonNegativeInt = Field(
        default=1,
        description="Number of repetitions of the model in the computed abcd matrix.",
    )

    def update(self, **kwargs) -> None:
        """
        Update multiple attributes of the model.

        Args:
            **kwargs: Attributes to update.
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise RuntimeError(f"The cell model does not have the {key} attribute.")

    def __mul__(self, factor: int) -> TwoPortModel:
        """
        Return model multiplied by integer factor.

        Args:
            factor (int): Multiplication factor.

        Returns:
            TwoPortModel: New instance of the model multiplied by the factor.
        """
        new_instance = self.__class__(**self.model_dump())
        new_instance.N = factor * self.N
        return new_instance

    @abstractmethod
    def single_abcd(self, freqs: np.ndarray) -> ABCDArray:
        """
        Compute the abcd matrix of a single iteration of the model.

        Args:
            freqs (numpy.ndarray): Frequencies array.

        Returns:
            ABCDArray: ABCD matrix for a single iteration.
        """

    def get_abcd(self, freqs: np.ndarray) -> ABCDArray:
        """
        Compute the abcd matrix of the model.

        Args:
            freqs (numpy.ndarray): Frequencies array.

        Returns:
            ABCDArray: Computed ABCD matrix of the model.
        """
        if self.N == 1:
            return self.single_abcd(freqs)
        return self.single_abcd(freqs) ** self.N

    def get_cell(self, freqs: np.ndarray) -> TwoPortCell:
        """
        Return the two-port cell of the model.

        Args:
            freqs (numpy.ndarray): Frequencies array.

        Returns:
            TwoPortCell: Two-port cell of the model.
        """
        return TwoPortCell(freqs, self.get_abcd(freqs), Z0=self.Z0_ref)

    def get_network(self, freqs: np.ndarray) -> rf.Network:
        """
        Return the two-port cell of the model as a scikit-rf Network.

        Args:
            freqs (numpy.ndarray): Frequencies array.

        Returns:
            rf.Network: scikit-rf Network representation of the model.
        """
        return self.get_cell(freqs).to_network()
