"""Models for TWPAS."""

from typing import List

import numpy as np
from pydantic import NonNegativeFloat

from twpasolver.abcd_matrices import ABCDArray, abcd_identity
from twpasolver.models.rf_functions import (
    LCLf_abcd,
    inductance,
    parallel_admittance_abcd,
    series_impedance_abcd,
    stub,
)
from twpasolver.twoport import TwoPortModel


class StubBaseCell(TwoPortModel):
    """Base cell of twpa stub filter model."""

    C: NonNegativeFloat
    L: NonNegativeFloat
    Lf: NonNegativeFloat
    l1: NonNegativeFloat
    l2: NonNegativeFloat
    n_stub: int = 2

    def single_abcd(self, freqs: np.ndarray) -> ABCDArray:
        """Compute abcd matrix."""
        stub_Y = self.n_stub * stub(freqs, self.Lf, self.C, 1)
        stub_abcd = ABCDArray(parallel_admittance_abcd(stub_Y))
        half_line_abcd = ABCDArray(series_impedance_abcd(inductance(freqs, self.L / 2)))
        return half_line_abcd @ stub_abcd @ half_line_abcd


class LCLfBaseCell(TwoPortModel):
    """Base cell of twpa LC model."""

    C: NonNegativeFloat
    L: NonNegativeFloat
    Lf: NonNegativeFloat

    def single_abcd(self, freqs: np.ndarray) -> ABCDArray:
        """Compute abcd matrix."""
        return ABCDArray(LCLf_abcd(freqs, self.C, self.L, self.Lf))


class TWPA(TwoPortModel):
    """Simple model for TWPAs."""

    cells: List[LCLfBaseCell | StubBaseCell]

    def single_abcd(self, freqs: np.ndarray) -> ABCDArray:
        """Compute abcd of supercell."""
        sc = abcd_identity(len(freqs))
        for cell in self.cells:
            sc = sc @ cell.get_abcd(freqs)
        return sc
