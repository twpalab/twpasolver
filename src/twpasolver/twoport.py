"""
Two-port RF networks: frequency-dependent ABCD matrices, conversion to S/Z parameters,
interpolation and file I/O.

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

import inspect
import itertools
from abc import ABC, abstractmethod
from typing import Any, ClassVar, cast

import numpy as np
import skrf as rf
from pydantic import ConfigDict, Field, NonNegativeInt

from twpasolver.basemodel import BaseModel
from twpasolver.bonus_types import Impedance, validate_impedance
from twpasolver.file_utils import read_file, save_to_file
from twpasolver.logger import log
from twpasolver.mathutils import a2s, a2z, s2a
from twpasolver.matrices_arrays import (
    ABCDArray,
    SMatrixArray,
    ZMatrixArray,
    abcd_identity,
)


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
            freqs (numpy.ndarray): Frequencies in Hz, shape (N,).
            abcd (numpy.ndarray | ABCDArray): ABCD matrices, shape (N, 2, 2).
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
            freqs (numpy.ndarray): Frequencies in Hz, shape (N,).
            s_mat (numpy.ndarray): S-parameter matrices, shape (N, 2, 2).
            Z0 (float | int): Reference line impedance.
        """
        abcd_mat = s2a(s_mat, Z0)
        return cls(freqs, abcd_mat, Z0=Z0)

    @classmethod
    def from_file(cls, filename: str, writer="hdf5") -> TwoPortCell:
        """Load cell from file written by :meth:`dump_to_file` (``writer`` is the file format)."""
        model_dict = read_file(filename, writer=writer)
        return cls(model_dict["freqs"], model_dict["abcd"], model_dict["Z0"])

    @property
    def freqs(self) -> np.ndarray:
        """Frequencies in Hz."""
        return self._freqs

    @freqs.setter
    def freqs(self, freqs: np.ndarray) -> None:
        """Set the frequencies (1-D, non-negative, same length as ``abcd``)."""
        if freqs.ndim != 1:
            raise ValueError("Frequencies must be 1-D array")
        if min(freqs) < 0:
            raise ValueError("Frequencies must be positive numbers")
        if freqs.shape[0] != self.abcd.shape[0]:
            raise ValueError("Frequencies and abcd matrices must have same length.")
        self._freqs = np.asarray(freqs)

    @property
    def Z0(self) -> Impedance:
        """Reference line impedance."""
        return self._Z0

    @Z0.setter
    def Z0(self, value: Impedance) -> None:
        """Set the reference line impedance."""
        validate_impedance(value)
        self._Z0 = value

    def to_network(self) -> rf.Network:
        """Convert to scikit-rf Network."""
        f = rf.Frequency.from_f(self.freqs * 1e-9, "ghz")
        # Pass S (pseudo-waves, as in ``self.s``), not ABCD: skrf converts ``a``
        # before ``z0`` is set.
        return rf.Network(frequency=f, s=np.asarray(self.s), z0=self.Z0, s_def="pseudo")

    @property
    def s(self) -> SMatrixArray:
        """S-parameter matrices (pseudo-waves, referenced to ``Z0``)."""
        return SMatrixArray(a2s(np.asarray(self.abcd), self.Z0))

    @property
    def z(self) -> ZMatrixArray:
        """Z-parameter matrices."""
        return ZMatrixArray(a2z(np.asarray(self.abcd)))

    def __repr__(self) -> str:
        """Return a string representation of the TwoPortCell."""
        return f"{self.__class__.__name__}(freqs={self.freqs},\nabcd={self.abcd},\nZ0={self.Z0})"

    def __getitem__(self, idxs: slice) -> TwoPortCell:
        """Slice the cell along frequency (only slices are allowed)."""
        if not isinstance(idxs, slice):
            raise ValueError("Only slicing of TwoPortCell is allowed")
        return self.__class__(self.freqs[idxs], self.abcd[idxs], self.Z0)

    def as_dict(self) -> dict:
        """Return cell contents as dictionary with keys ``freqs``, ``abcd``, ``Z0``."""
        return {"freqs": self.freqs, "abcd": np.asarray(self.abcd), "Z0": self.Z0}

    def dump_to_file(self, filename: str, writer="hdf5") -> None:
        """Dump cell to file (``writer`` is the file format)."""
        save_to_file(filename, self.as_dict(), writer=writer)

    def interpolate(self, freqs: np.ndarray, polar: bool = False) -> TwoPortCell:
        """
        Return a cell with ABCD matrices linearly interpolated at the given frequencies.

        Args:
            freqs (numpy.ndarray): Frequencies in Hz.
            polar (bool): Interpolate magnitude and unwrapped phase instead of real and imaginary part.
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

    # Models declaring their own ``name`` field, by name, for validating cells from dicts.
    registry: ClassVar[dict[str, type[TwoPortModel]]] = {}

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs: Any) -> None:
        """Register a subclass that declares its own ``name`` field (default = key)."""
        super().__pydantic_init_subclass__(**kwargs)
        if "name" in inspect.get_annotations(cls):
            TwoPortModel.registry[cls.model_fields["name"].default] = cls

    @staticmethod
    def validate_cell(cell: Any, base: type) -> TwoPortModel:
        """
        A copy of ``cell`` (a model instance, or a dict with a registered ``name``),
        checked to be a ``base``.
        """
        if isinstance(cell, dict):
            model = TwoPortModel.registry.get(cell.get("name", ""))
            if model is None:
                raise ValueError(
                    f"Unknown model name {cell.get('name')!r}; registered: "
                    f"{sorted(TwoPortModel.registry)}."
                )
            cell = model.model_validate(cell)
        elif isinstance(cell, TwoPortModel):
            cell = type(cell).model_validate(cell)
        if not isinstance(cell, base):
            raise ValueError(f"Expected a {base.__name__}, got {type(cell).__name__}.")
        return cast(TwoPortModel, cell)

    def update(self, **kwargs) -> None:
        """Update multiple attributes of the model; unknown names raise RuntimeError."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise RuntimeError(f"The cell model does not have the {key} attribute.")

    def __mul__(self, factor: int) -> TwoPortModel:
        """Return a copy of the model with ``N`` multiplied by ``factor``."""
        new = self.model_copy(deep=True)
        new.N = factor * self.N
        return new

    @abstractmethod
    def single_abcd(self, freqs: np.ndarray) -> ABCDArray:
        """
        Compute the ABCD matrix of a single iteration of the model.

        Args:
            freqs (numpy.ndarray): Frequencies in Hz.
        """

    def get_abcd(self, freqs: np.ndarray) -> ABCDArray:
        """Compute the ABCD matrix of the model repeated ``N`` times; freqs in Hz."""
        if self.N == 0:
            return abcd_identity(len(freqs))
        if self.N == 1:
            return self.single_abcd(freqs)
        return self.single_abcd(freqs) ** self.N

    def get_cell(self, freqs: np.ndarray) -> TwoPortCell:
        """Return the two-port cell of the model; freqs in Hz."""
        return TwoPortCell(freqs, self.get_abcd(freqs), Z0=self.Z0_ref)

    def get_network(self, freqs: np.ndarray) -> rf.Network:
        """Return the two-port cell of the model as a scikit-rf Network; freqs in Hz."""
        return self.get_cell(freqs).to_network()
