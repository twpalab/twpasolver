"""
Module containing models for transmission lines.

This module provides classes and functions to model various types of transmission lines.
"""

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
        Initialize the model from impedance and line propagation constant.

        Args:
            Z0 (NonNegativeFloat): Characteristic impedance of the line.
            vp (NonNegativeFloat): Phase velocity of the line.
            l (NonNegativeFloat): Length of the transmission line.

        Returns:
            LosslessTL: An instance of the LosslessTL class.
        """
        return cls(l=l, L=Z0 / vp, C=1 / (vp * Z0))

    @computed_field  # type: ignore
    @property
    def Z0(self) -> NonNegativeFloat:
        """
        Compute the characteristic impedance of the line.

        Returns:
            NonNegativeFloat: The characteristic impedance.
        """
        return np.sqrt(self.L / self.C)

    @computed_field  # type: ignore
    @property
    def vp(self) -> NonNegativeFloat:
        """
        Compute the characteristic phase velocity of the line.

        Returns:
            NonNegativeFloat: The phase velocity.
        """
        return 1 / np.sqrt(self.L * self.C)

    def single_abcd(self, freqs: np.ndarray) -> ABCDArray:
        """
        Get the ABCD matrix of the transmission line portion for given frequencies.

        Args:
            freqs (np.ndarray): Array of frequencies at which to evaluate the ABCD matrix.

        Returns:
            ABCDArray: The ABCD matrix of the transmission line.
        """
        return ABCDArray(lossless_line_abcd(freqs, self.C, self.L, self.l))
