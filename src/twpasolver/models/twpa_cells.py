"""Models for TWPAS."""

from __future__ import annotations

from functools import partial
from typing import Literal

import numpy as np
from pydantic import Field, NonNegativeFloat

from twpasolver.matrices_arrays import ABCDArray
from twpasolver.models.rf_functions import (
    LCLf_abcd,
    get_stub_cell,
    inductance,
    series_impedance_abcd,
)
from twpasolver.twoport import TwoPortModel

required = partial(Field, ...)


class StubBaseCell(TwoPortModel):
    """
    Base cell of the twpa stub-filter model: a centre line loaded by two open stubs.

    ``line=True``: lumped centre line split around the stubs, ``[L/2][stubs][L/2]``.
    Each of the two open stubs is a distributed line of length ``l1`` with inductance
    ``Lf`` and capacitance ``C`` per unit length (total per stub ``Lf*l1``, ``C*l1``;
    shunt admittance ``2j tan(w sqrt(Lf C) l1) / sqrt(Lf/C)``). Below the stub resonance
    this is the centered ``LCLfBaseCell`` with total capacitance ``2 C l1`` and finger
    inductance ``Lf l1 / 3`` (same expansion to order w^3). ``qp_loss`` acts on the centre
    line, ``delta`` on the stub capacitance. Supported by the Bloch CMEs (the two halves
    of ``L`` are the nonlinear elements).

    ``line=False`` (legacy): stubs and centre line share ``L`` and ``C`` per unit length,
    centre line lumped (``L*l2``) after the stubs; ``Lf`` and ``qp_loss`` are unused.
    """

    name: Literal["StubBaseCell"] = "StubBaseCell"
    L: NonNegativeFloat = required(description="Inductance of the straight line.")
    C: NonNegativeFloat = required(
        description="Capacitance per unit length of each stub."
    )
    Lf: NonNegativeFloat = required(
        description="Inductance per unit length of each stub (line=True)."
    )
    l1: NonNegativeFloat = Field(1, description="Length of the stub finger.")
    l2: NonNegativeFloat = Field(1, description="Length of the straight line.")
    qp_loss: NonNegativeFloat = Field(0, description=("Quasiparticle-loss coefficient"))
    delta: NonNegativeFloat = Field(0, description="Loss Tangent")
    line: bool = Field(
        False,
        description=(
            "Lumped centre line split around stubs with their own inductance Lf "
            "([L/2][stubs][L/2]) instead of the legacy shared-L stub cell."
        ),
    )

    def single_abcd(self, freqs: np.ndarray) -> ABCDArray:
        """Compute abcd matrix."""
        if self.line:
            parallel_stubs = ABCDArray(
                get_stub_cell(
                    freqs, self.C * (1 - 1j * self.delta), self.Lf, self.l1, 0
                )
            )
            half_Z = inductance(freqs, self.L / 2) * (1 - 1j * self.qp_loss * freqs)
            half_inductance = ABCDArray(series_impedance_abcd(half_Z))
            return half_inductance @ parallel_stubs @ half_inductance

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
    qp_loss: NonNegativeFloat = Field(0, description=("Quasiparticle-loss coefficient"))
    centered: bool = False

    def single_abcd(self, freqs: np.ndarray) -> ABCDArray:
        """Compute abcd matrix."""
        if self.centered:
            fingers = ABCDArray(
                LCLf_abcd(freqs, self.C * (1 - 1j * self.delta), 0, self.Lf)
            )
            half_Z = inductance(freqs, self.L / 2) * (1 - 1j * self.qp_loss * freqs)
            half_inductance = ABCDArray(series_impedance_abcd(half_Z))
            return half_inductance @ fingers @ half_inductance

        return ABCDArray(
            LCLf_abcd(
                freqs,
                self.C * (1 - 1j * self.delta),
                self.L,
                self.Lf,
                self.qp_loss,
            )
        )
