"""
Generic models for user-defined abcd matrices and TwoPortModel lists.

This module provides a generic container class for arrays of TwoPortModels, allowing for the
organization and manipulation of multiple two-port network models, including nested lists.
It also includes specific models like TWPA for specialized applications.
"""

# mypy: ignore-errors
from __future__ import annotations

from typing import Annotated, Literal, Union

import numpy as np
from pydantic import Field, NonNegativeFloat, NonNegativeInt, computed_field

from twpasolver.bonus_types import all_subclasses
from twpasolver.matrices_arrays import ABCDArray, abcd_identity
from twpasolver.models.modelarray import ModelArray
from twpasolver.twoport import TwoPortModel


class TwoPortArray(ModelArray):
    """Generic container for arrays of TwoPortModels."""

    name: Literal["TwoPortArray"] = Field(default="TwoPortArray")
    cells: list[AnyModel] = Field(
        default_factory=list,
        description="List of TwoPortModels representing the basic cells. Nested lists are allowed.",
    )

    def single_abcd(self, freqs: np.ndarray) -> ABCDArray:
        """
        Compute abcd of combined models.

        Args:
            freqs (np.ndarray): Array of frequencies.

        Returns:
            ABCDArray: The combined ABCD matrix of the models.
        """
        sc = abcd_identity(len(freqs))
        for cell in self.cells:
            sc = sc @ cell.get_abcd(freqs)
        return sc

    @computed_field
    @property
    def N_tot(self) -> NonNegativeInt:
        """
        Total number of base cells in the model.

        Returns:
            NonNegativeInt: The total number of base cells.
        """
        N_tot = 0
        for cell in self.cells:
            if hasattr(cell, "N_tot"):
                N_tot += cell.N_tot
            else:
                N_tot += cell.N
        return N_tot * self.N


class TWPA(TwoPortArray):
    """Simple model for TWPAs."""

    name: Literal["TWPA"] = Field(default="TWPA")
    Istar: NonNegativeFloat = Field(
        default=6.5e-3, description="Nonlinearity scale current parameter."
    )
    Idc: NonNegativeFloat = Field(default=1e-3, description="Bias dc current.")
    Ip0: NonNegativeFloat = Field(default=2e-4, description="Rf pump amplitude.")

    @computed_field
    @property
    def epsilon(self) -> NonNegativeFloat:
        """
        Coefficient of first-order term in inductance.

        Returns:
            NonNegativeFloat: The epsilon value.
        """
        return 2 * self.Idc / (self.Idc**2 + self.Istar**2)

    @computed_field
    @property
    def xi(self) -> NonNegativeFloat:
        """
        Coefficient of second-order term in inductance.

        Returns:
            NonNegativeFloat: The xi value.
        """
        return 1 / (self.Idc**2 + self.Istar**2)

    @computed_field
    @property
    def chi(self) -> NonNegativeFloat:
        """
        Coefficient for second term of phase matching relation.

        Returns:
            NonNegativeFloat: The chi value.
        """
        return self.Ip0**2 * self.xi / 8

    @computed_field
    @property
    def alpha(self) -> NonNegativeFloat:
        """
        Second-order correction term for inductance as function of dc current.

        Returns:
            NonNegativeFloat: The alpha value.
        """
        return 1 + self.Idc**2 / self.Istar**2

    @computed_field
    @property
    def Iratio(self) -> NonNegativeFloat:
        """
        Ratio between bias and nonlinearity current parameter.

        Returns:
            NonNegativeFloat: The Iratio value.
        """
        return self.Idc / self.Istar


named_models = [c for c in all_subclasses(TwoPortModel) if "name" in c.model_fields]
AnyModel = Annotated[Union[tuple(named_models)], Field(discriminator="name")]
