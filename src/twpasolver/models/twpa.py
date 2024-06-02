"""Models for TWPAS."""

from __future__ import annotations

from functools import partial

import numpy as np
from pydantic import Field, NonNegativeFloat, NonNegativeInt, computed_field

from twpasolver.abcd_matrices import ABCDArray, abcd_identity
from twpasolver.models.base_components import Inductance
from twpasolver.models.rf_functions import LCLf_abcd, get_stub_cell, lossless_line_abcd
from twpasolver.twoport import TwoPortModel

required = partial(Field, ...)


class StubBaseCell(TwoPortModel):
    """Base cell of twpa stub filter model."""

    L: NonNegativeFloat = required(description="Inductance of the straight line.")
    C: NonNegativeFloat = required(description="Capacitance of the line.")
    Lf: NonNegativeFloat = required(description="Inductance of the stub finger.")
    l1: NonNegativeFloat = required(description="Length of the stub finger.")
    l2: NonNegativeFloat = required(description="Length of the straight line.")
    line: bool = Field(
        False,
        description="Model line as distributed element instead of lumped inductance.",
    )

    def single_abcd(self, freqs: np.ndarray) -> ABCDArray:
        """Compute abcd matrix."""
        if self.line:
            parallel_stubs = ABCDArray(
                -1 * get_stub_cell(freqs, self.C, self.Lf, self.l1, 0)
            )
            half_line = ABCDArray(
                lossless_line_abcd(freqs, self.C, self.L, self.l2 / 2)
            )
            return half_line @ parallel_stubs @ half_line

        return ABCDArray(get_stub_cell(freqs, self.C, self.L, self.l1, self.l2))


class LCLfBaseCell(TwoPortModel):
    """Base cell of twpa LC model."""

    L: NonNegativeFloat = required(description="Inductance of the straight line.")
    C: NonNegativeFloat = required(description="Capacitance of the stub finger.")
    Lf: NonNegativeFloat = required(description="Inductance of the stub finger.")
    centered: bool = False

    def single_abcd(self, freqs: np.ndarray) -> ABCDArray:
        """Compute abcd matrix."""
        if self.centered:
            fingers = ABCDArray(LCLf_abcd(freqs, self.C, 0, self.Lf))
            half_inductance = Inductance(L=self.L / 2).get_abcd(freqs)
            return half_inductance @ fingers @ half_inductance
        return ABCDArray(LCLf_abcd(freqs, self.C, self.L, self.Lf))


class TWPA(TwoPortModel):
    """Simple model for TWPAs."""

    cells: list[LCLfBaseCell | StubBaseCell] = required(
        description="List of TwoPortModels representing the basic cells."
    )
    Istar: NonNegativeFloat = Field(
        default=6.5e-3, description="Nonlinearity scale current parameter."
    )
    Idc: NonNegativeFloat = Field(default=1e-3, description="Bias dc current.")
    Ip0: NonNegativeFloat = Field(default=2e-4, description="Rf pump amplitude.")

    def single_abcd(self, freqs: np.ndarray) -> ABCDArray:
        """Compute abcd of supercell."""
        sc = abcd_identity(len(freqs))
        for cell in self.cells:
            sc = sc @ cell.get_abcd(freqs)
        return sc

    @computed_field  # type: ignore[misc]
    @property
    def epsilon(self) -> NonNegativeFloat:
        """Coefficient of first-order term in inductance."""
        return 2 * self.Idc / (self.Idc**2 + self.Istar**2)

    @computed_field  # type: ignore[misc]
    @property
    def xi(self) -> NonNegativeFloat:
        """Coefficient of second-order term in inductance."""
        return 1 / (self.Idc**2 + self.Istar**2)

    @computed_field  # type: ignore[misc]
    @property
    def chi(self) -> NonNegativeFloat:
        """Coefficient for second term of phase matching relation."""
        return self.Ip0**2 * self.xi / 8

    @computed_field  # type: ignore[misc]
    @property
    def alpha(self) -> NonNegativeFloat:
        """Second-order correction term for inductance as function of dc current."""
        return 1 + self.Idc**2 / self.Istar**2

    @computed_field  # type: ignore[misc]
    @property
    def Iratio(self) -> NonNegativeFloat:
        """Ratio between bias and nonlinearity current parameter."""
        return self.Idc / self.Istar

    @computed_field  # type: ignore[misc]
    @property
    def N_tot(self) -> NonNegativeInt:
        """Total number of base cells in the model."""
        return sum([cell.N for cell in self.cells]) * self.N
