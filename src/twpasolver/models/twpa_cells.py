"""Models for TWPAS."""

from __future__ import annotations

from functools import partial
from typing import Literal

import numpy as np
from pydantic import Field, NonNegativeFloat

from twpasolver.matrices_arrays import ABCDArray
from twpasolver.models.oneport import Inductance
from twpasolver.models.rf_functions import LCLf_abcd, get_stub_cell, lossless_line_abcd
from twpasolver.twoport import TwoPortModel

required = partial(Field, ...)


class StubBaseCell(TwoPortModel):
    """Base cell of twpa stub filter model."""

    name: Literal["StubBaseCell2"] = "StubBaseCell2"
    L: NonNegativeFloat = required(description="Inductance of the straight line.")
    C: NonNegativeFloat = required(description="Capacitance of the line.")
    Lf: NonNegativeFloat = required(description="Inductance of the stub finger.")
    l1: NonNegativeFloat = required(description="Length of the stub finger.")
    l2: NonNegativeFloat = required(description="Length of the straight line.")
    delta: NonNegativeFloat = Field(0, description="Loss Tangent")
    line: bool = Field(
        False,
        description="Model line as distributed element instead of lumped inductance.",
    )

    def single_abcd(self, freqs: np.ndarray) -> ABCDArray:
        """Compute abcd matrix."""
        if self.line:
            parallel_stubs = ABCDArray(
                -1
                * get_stub_cell(
                    freqs, self.C * (1 - 1j * self.delta), self.Lf, self.l1, 0
                )
            )
            half_line = ABCDArray(
                lossless_line_abcd(
                    freqs, self.C * (1 - 1j * self.delta), self.L, self.l2 / 2
                )
            )
            return half_line @ parallel_stubs @ half_line

        return ABCDArray(
            get_stub_cell(
                freqs, self.C * (1 - 1j * self.delta), self.L, self.l1, self.l2
            )
        )


class LCLfBaseCell(TwoPortModel):
    """Base cell of twpa LC model."""

    name: Literal["LCLfBaseCell"] = "LCLfBaseCell"
    L: NonNegativeFloat = required(description="Inductance of the straight line.")
    C: NonNegativeFloat = required(description="Capacitance of the stub finger.")
    Lf: NonNegativeFloat = required(description="Inductance of the stub finger.")
    delta: NonNegativeFloat = Field(0, description="Loss Tangent")
    centered: bool = False

    def single_abcd(self, freqs: np.ndarray) -> ABCDArray:
        """Compute abcd matrix."""
        if self.centered:
            fingers = ABCDArray(
                LCLf_abcd(freqs, self.C * (1 - 1j * self.delta), 0, self.Lf)
            )
            half_inductance = Inductance(L=self.L / 2).get_abcd(freqs)
            return half_inductance @ fingers @ half_inductance
        return ABCDArray(
            LCLf_abcd(freqs, self.C * (1 - 1j * self.delta), self.L, self.Lf)
        )
