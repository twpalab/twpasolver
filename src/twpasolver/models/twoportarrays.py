"""Arrays of TwoPortModels, including nested lists, and the TWPA model."""

from __future__ import annotations

from typing import Literal

import numpy as np
from pydantic import Field, NonNegativeFloat, NonNegativeInt, computed_field

from twpasolver.matrices_arrays import ABCDArray, abcd_identity
from twpasolver.models.modelarray import ModelArray


class TwoPortArray(ModelArray):
    """Generic container for arrays of TwoPortModels."""

    name: Literal["TwoPortArray"] = Field(default="TwoPortArray")

    def single_abcd(self, freqs: np.ndarray) -> ABCDArray:
        """Cascade the ABCD matrices of the cells."""
        sc = abcd_identity(len(freqs))
        for cell in self.cells:
            sc = sc @ cell.get_abcd(freqs)
        return sc

    @computed_field  # type: ignore[prop-decorator]
    @property
    def N_tot(self) -> NonNegativeInt:
        """Total number of base cells, counting nested arrays and repetitions ``N``."""
        N_tot = 0
        for cell in self.cells:
            if hasattr(cell, "N_tot"):
                N_tot += cell.N_tot
            else:
                N_tot += cell.N
        return N_tot * self.N


class TWPA(TwoPortArray):
    """Simple model for TWPAs."""

    name: Literal["TWPA"] = Field(default="TWPA")  # type: ignore[assignment]
    Istar: NonNegativeFloat = Field(
        default=6.5e-3, description="Nonlinearity scale current parameter."
    )
    Istar2: NonNegativeFloat | None = Field(
        None, description="Second order scaling current parameter."
    )
    Idc: NonNegativeFloat = Field(default=1e-3, description="Bias dc current.")
    Ip0: NonNegativeFloat = Field(default=2e-4, description="Rf pump amplitude.")

    @computed_field  # type: ignore[prop-decorator]
    @property
    def epsilon(self) -> NonNegativeFloat:
        """
        Coefficient of first-order term in inductance.

        With L(I) = Lk (1 + (I/Istar)^2 + (I/Istar2)^4) and I = Idc + i,
        L = L(Idc) (1 + epsilon i + xi i^2 + O(i^3)).
        """
        if self.Istar2 is None:
            return 2 * self.Idc * self.xi

        return (
            2
            * self.Idc
            * (2 * (self.Istar * self.Idc) ** 2 + self.Istar2**4)
            / self._xi_denominator
        )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def xi(self) -> NonNegativeFloat:
        """Coefficient of second-order term in inductance (see ``epsilon``)."""

        if self.Istar2 is None:
            return 1 / (self.Istar**2 + self.Idc**2)

        return (
            6 * (self.Istar * self.Idc) ** 2 + self.Istar2**4
        ) / self._xi_denominator

    @property
    def _xi_denominator(self) -> float:
        """
        Common denominator of epsilon and xi when Istar2 is set.

        Equals Istar^2 * Istar2^4 * L(Idc) / Lk.
        """
        assert self.Istar2 is not None
        return (
            (self.Istar * self.Istar2**2) ** 2
            + (self.Istar * self.Idc**2) ** 2
            + (self.Idc * self.Istar2**2) ** 2
        )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def chi(self) -> NonNegativeFloat:
        """Coefficient for second term of phase matching relation."""
        return self.Ip0**2 * self.xi / 8

    @computed_field  # type: ignore[prop-decorator]
    @property
    def alpha(self) -> NonNegativeFloat:
        """Second-order correction term for inductance as function of dc current."""
        alpha = 1 + self.Idc**2 / self.Istar**2
        if self.Istar2 is not None:
            alpha += self.Idc**4 / self.Istar2**4
        return alpha

    @computed_field  # type: ignore[prop-decorator]
    @property
    def Iratio(self) -> NonNegativeFloat:
        """Ratio ``Idc / Istar`` between bias and nonlinearity current."""
        return self.Idc / self.Istar
