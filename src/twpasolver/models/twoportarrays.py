"""Generic models for user-defined abcd matrices and TwoPortModel lists."""

# mypy: ignore-errors
from __future__ import annotations

from typing import Annotated, Literal, Union

import numpy as np
from pydantic import Field, NonNegativeFloat, NonNegativeInt, computed_field

from twpasolver.matrices_arrays import ABCDArray, abcd_identity
from twpasolver.models.modelarray import ModelArray
from twpasolver.twoport import TwoPortModel


def all_subclasses(cls):
    """Recursively get all subclasses of a given class."""
    return cls.__subclasses__() + [
        s for c in cls.__subclasses__() for s in all_subclasses(c)
    ]


class TwoPortArray(ModelArray):
    """Generic container for arrays of TwoPortModels."""

    name: Literal["TwoPortArray"] = Field(default="TwoPortArray")
    cells: list[AnyModel] = Field(
        default_factory=list,
        description="List of TwoPortModels representing the basic cells. Nested lists are allowed.",
    )

    def single_abcd(self, freqs: np.ndarray) -> ABCDArray:
        """Compute abcd of combined models."""
        sc = abcd_identity(len(freqs))
        for cell in self.cells:
            sc = sc @ cell.get_abcd(freqs)
        return sc


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
        """Coefficient of first-order term in inductance."""
        return 2 * self.Idc / (self.Idc**2 + self.Istar**2)

    @computed_field
    @property
    def xi(self) -> NonNegativeFloat:
        """Coefficient of second-order term in inductance."""
        return 1 / (self.Idc**2 + self.Istar**2)

    @computed_field
    @property
    def chi(self) -> NonNegativeFloat:
        """Coefficient for second term of phase matching relation."""
        return self.Ip0**2 * self.xi / 8

    @computed_field
    @property
    def alpha(self) -> NonNegativeFloat:
        """Second-order correction term for inductance as function of dc current."""
        return 1 + self.Idc**2 / self.Istar**2

    @computed_field
    @property
    def Iratio(self) -> NonNegativeFloat:
        """Ratio between bias and nonlinearity current parameter."""
        return self.Idc / self.Istar

    @computed_field
    @property
    def N_tot(self) -> NonNegativeInt:
        """Total number of base cells in the model."""
        return sum(cell.N for cell in self.cells) * self.N


named_models = [c for c in all_subclasses(TwoPortModel) if "name" in c.model_fields]
AnyModel = Annotated[Union[tuple(named_models)], Field(discriminator="name")]
