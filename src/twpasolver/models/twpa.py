"""Models for TWPAS."""

from functools import partial
from typing import List, Optional

import numpy as np
from pydantic import Field, NonNegativeFloat

from twpasolver.abcd_matrices import ABCDArray, abcd_identity
from twpasolver.models.rf_functions import (
    LCLf_abcd,
    inductance,
    lossless_line_abcd,
    parallel_admittance_abcd,
    series_impedance_abcd,
    stub,
)
from twpasolver.twoport import TwoPortModel

required = partial(Field, ...)


class StubBaseCell(TwoPortModel):
    """Base cell of twpa stub filter model."""

    L: NonNegativeFloat = required(description="Inductance of the straight line.")
    C: NonNegativeFloat = required(description="Capacitance of the stub finger.")
    Lf: NonNegativeFloat = required(description="Capacitance of the stub finger.")
    l1: NonNegativeFloat = required(description="Length of the stub finger.")
    l2: NonNegativeFloat = required(description="Length of the straight line.")
    n_stub: int = Field(2, description="Number of parallel stub fingers.")
    line: bool = Field(
        False,
        description="Model line as distributed element instead of lumped inductance.",
    )

    def single_abcd(self, freqs: np.ndarray) -> ABCDArray:
        """Compute abcd matrix."""
        stub_Z = stub(freqs, self.Lf, self.C, 1)
        stub_abcd = ABCDArray(parallel_admittance_abcd(self.n_stub / stub_Z))
        if self.line:
            half_line_abcd = ABCDArray(
                lossless_line_abcd(freqs, self.C / self.l1, self.Lf / self.l1, self.l2)
            )
        else:
            half_line_abcd = ABCDArray(
                series_impedance_abcd(inductance(freqs, self.L / 2))
            )
        return half_line_abcd @ stub_abcd @ half_line_abcd


class LCLfBaseCell(TwoPortModel):
    """Base cell of twpa LC model."""

    L: NonNegativeFloat = required(description="Inductance of the straight line.")
    C: NonNegativeFloat = required(description="Capacitance of the stub finger.")
    Lf: NonNegativeFloat = required(description="Inductance of the stub finger.")

    def single_abcd(self, freqs: np.ndarray) -> ABCDArray:
        """Compute abcd matrix."""
        return ABCDArray(LCLf_abcd(freqs, self.C, self.L, self.Lf))


class TWPA(TwoPortModel):
    """Simple model for TWPAs."""

    cells: List[LCLfBaseCell | StubBaseCell] = required(
        description="List of TwoPortModels representing the basic cells."
    )
    Istar: Optional[NonNegativeFloat] = None
    Ic: Optional[NonNegativeFloat] = None
    Ip0: Optional[NonNegativeFloat] = None
    Is0: Optional[NonNegativeFloat] = None

    def single_abcd(self, freqs: np.ndarray) -> ABCDArray:
        """Compute abcd of supercell."""
        sc = abcd_identity(len(freqs))
        for cell in self.cells:
            sc = sc @ cell.get_abcd(freqs)
        return sc
