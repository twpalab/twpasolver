"""Transmission line models."""

from typing import Literal

import numpy as np
from pydantic import Field, NonNegativeFloat, computed_field

from twpasolver.matrices_arrays import ABCDArray
from twpasolver.models.rf_functions import lossless_line_abcd
from twpasolver.twoport import TwoPortModel


class LosslessTL(TwoPortModel):
    """Model for a lossless transmission line."""

    name: Literal["LosslessTL"] = "LosslessTL"
    l: NonNegativeFloat = Field(..., description="Length of the line.")
    L: NonNegativeFloat = Field(
        ..., description="Characteristic inductance of the line."
    )
    C: NonNegativeFloat = Field(
        ..., description="Characteristic capacitance of the line."
    )

    @classmethod
    def from_z_vp(
        cls, Z0: NonNegativeFloat, vp: NonNegativeFloat, l: NonNegativeFloat
    ) -> "LosslessTL":
        """
        Build the line from characteristic impedance ``Z0``, phase velocity ``vp`` and length ``l``.
        """
        return cls(l=l, L=Z0 / vp, C=1 / (vp * Z0))

    @computed_field  # type: ignore
    @property
    def Z0(self) -> NonNegativeFloat:
        """Characteristic impedance ``sqrt(L / C)``."""
        return np.sqrt(self.L / self.C)

    @computed_field  # type: ignore
    @property
    def vp(self) -> NonNegativeFloat:
        """Phase velocity ``1 / sqrt(L C)``."""
        return 1 / np.sqrt(self.L * self.C)

    def single_abcd(self, freqs: np.ndarray) -> ABCDArray:
        """ABCD matrices of the line."""
        return ABCDArray(lossless_line_abcd(freqs, self.C, self.L, self.l))
