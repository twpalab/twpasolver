"""Simple components."""

from abc import ABC, abstractmethod
from functools import partial
from typing import Callable, Literal

import numpy as np
from pydantic import NonNegativeFloat, PrivateAttr

from twpasolver.matrices_arrays import ABCDArray
from twpasolver.models.rf_functions import (
    capacitance,
    inductance,
    parallel_admittance_abcd,
    series_impedance_abcd,
    stub,
)
from twpasolver.twoport import TwoPortModel
from twpasolver.typing import complex_array, float_array


class Component(TwoPortModel, ABC):
    """Base class for single components."""

    parallel: bool = False
    _get_impedance: Callable[[float_array], complex_array] = PrivateAttr()

    @abstractmethod
    def model_post_init(self, _):
        """Set _get_impedance function here."""

    def single_abcd(self, freqs: np.ndarray) -> ABCDArray:
        """Get abcd of series impedance or parallel admittance."""
        impedance = self._get_impedance(freqs)
        if self.parallel:
            return ABCDArray(parallel_admittance_abcd(1 / impedance))
        return ABCDArray(series_impedance_abcd(impedance))


class Capacitance(Component):
    """Model of a capacitor."""

    name: Literal["Capacitance"] = "Capacitance"
    C: NonNegativeFloat

    def model_post_init(self, _):
        """Set _get_impedance function here."""
        self._get_impedance = partial(capacitance, C=self.C)


class Inductance(Component):
    """Model of an inductance."""

    name: Literal["Inductance"] = "Inductance"
    L: NonNegativeFloat

    def model_post_init(self, _):
        """Set _get_impedance function here."""
        self._get_impedance = partial(inductance, L=self.L)


class Stub(Component):
    """Model of a stub."""

    name: Literal["Stub"] = "Stub"
    L: NonNegativeFloat
    C: NonNegativeFloat
    length: NonNegativeFloat
    open: bool = True

    def model_post_init(self, _):
        """Set _get_impedance function here."""
        self._get_impedance = partial(
            stub, L=self.L, C=self.C, length=self.length, open=self.open
        )
