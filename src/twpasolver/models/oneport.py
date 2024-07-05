"""Simple components."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Annotated, Literal, Union

import numpy as np
from pydantic import Field, NonNegativeFloat

from twpasolver.matrices_arrays import ABCDArray
from twpasolver.models.modelarray import ModelArray
from twpasolver.models.rf_functions import (
    capacitance,
    inductance,
    parallel_admittance_abcd,
    series_impedance_abcd,
    stub,
)
from twpasolver.twoport import TwoPortModel


class OnePortModel(TwoPortModel, ABC):
    """Base class for components entirely described by an 1D impedance array."""

    to_twoport_as_parallel: bool = Field(
        default=False,
        description="Insert component in parallel when transforming to twoport representation.",
    )

    @abstractmethod
    def Z(self, freqs: np.ndarray) -> np.ndarray:
        """Get impedance of component as funcion of frequency."""

    def Y(self, freqs: np.ndarray) -> np.ndarray:
        """Get admittance of component as funcion of frequency."""
        return 1 / self.Z(freqs)

    def single_abcd(self, freqs: np.ndarray) -> ABCDArray:
        """Get abcd of series impedance or parallel admittance."""
        if self.to_twoport_as_parallel:
            return ABCDArray(parallel_admittance_abcd(self.Y(freqs)))
        return ABCDArray(series_impedance_abcd(self.Z(freqs)))


class OnePortArray(OnePortModel, ModelArray):
    """Container for direct compositions of OnePortModels."""

    name: Literal["OnePortArray"] = Field(default="OnePortArray")
    parallel: bool = Field(
        default=False, description="Connect internal OnePortModels in parallel."
    )
    cells: list[AnyOnePortModel] = Field(
        default_factory=list,
        description="List of OnePortModel representing the basic cells. Nested cells are allowed.",
    )

    def Z(self, freqs: np.ndarray) -> np.ndarray:
        """Get impedance of the composed elements."""
        if self.parallel:
            return 1 / np.sum([c.Y(freqs) for c in self.cells], axis=0)
        return np.sum([c.Z(freqs) for c in self.cells], axis=0)


class Resistance(OnePortModel):
    """Model of a resistor."""

    name: Literal["Resistance"] = "Resistance"
    R: NonNegativeFloat

    def Z(self, freqs: np.ndarray) -> np.ndarray:
        """Get impedance of resistor."""
        return np.full_like(freqs, self.R)


class Capacitance(OnePortModel):
    """Model of a capacitor."""

    name: Literal["Capacitance"] = "Capacitance"
    C: NonNegativeFloat

    def Z(self, freqs: np.ndarray) -> np.ndarray:
        """Get impedance of capacitor."""
        return capacitance(freqs, self.C)


class Inductance(OnePortModel):
    """Model of an inductance."""

    name: Literal["Inductance"] = "Inductance"
    L: NonNegativeFloat

    def Z(self, freqs: np.ndarray) -> np.ndarray:
        """Get impedance of inductor."""
        return inductance(freqs, self.L)


class Stub(OnePortModel):
    """Model of a stub."""

    name: Literal["Stub"] = "Stub"
    L: NonNegativeFloat
    C: NonNegativeFloat
    length: NonNegativeFloat
    open: bool = True

    def Z(self, freqs: np.ndarray) -> np.ndarray:
        """Get impedance of stub."""
        return stub(freqs, L=self.L, C=self.C, length=self.length, open=self.open)


AnyOnePortModel = Annotated[
    Union[Resistance, Inductance, Capacitance, Stub, OnePortArray],
    Field(discriminator="name"),
]
