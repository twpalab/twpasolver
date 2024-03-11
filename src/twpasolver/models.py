"""ABCD matrices models module."""
from typing import Any

import numpy as np
from numba import njit
from pydantic import NonNegativeFloat, NonNegativeInt

from twpasolver.abcd_matrices import ABCDArray
from twpasolver.twoport import TwoPortModel
from twpasolver.typing import complex_array


@njit
def get_malnou_base(
    freqs: np.ndarray[Any, np.dtype[np.float64]], C: float, L: float, Lf: float
) -> complex_array:
    """
    Get base abcd matrix of Malnou model.

    'ABCD matrix computation for the single cell of a fishbone line
    PRX Quantum 2 (2021) 010302
    https://doi.org/10.1103/PRXQuantum.2.010302'
    """
    assert C >= 0
    assert L >= 0
    assert Lf >= 0
    n_mat = len(freqs)
    abcd = np.empty((n_mat, 2, 2), dtype=np.complex128)
    for i in range(n_mat):
        w = 2 * np.pi * freqs[i]
        den = 2 - Lf * C * w**2
        abcd[i, 0, 0] = 1
        abcd[i, 0, 1] = 1j * L * w
        abcd[i, 1, 0] = 1j * 2 * C * w / den
        abcd[i, 1, 1] = 1 - 2 * L * C * w**2 / den
    return abcd


class MalnouBaseCell(TwoPortModel):
    """Base cell of Malnou model."""

    C: NonNegativeFloat
    L: NonNegativeFloat
    Lf: NonNegativeFloat

    def get_abcd(self, freqs: np.ndarray) -> ABCDArray:
        """Compute abcd matrix."""
        return ABCDArray(get_malnou_base(freqs, self.C, self.L, self.Lf))


class TWPA(TwoPortModel):
    """Simple model for TWPAs."""

    unloaded: MalnouBaseCell
    loaded: MalnouBaseCell
    N_l: NonNegativeInt
    N_u: NonNegativeInt
    N_sc: NonNegativeInt

    def get_abcd(self, freqs: np.ndarray) -> np.ndarray | ABCDArray:
        """Compute abcd."""
        abcd_u = self.unloaded.get_abcd(freqs)
        abcd_l = self.loaded.get_abcd(freqs)
        u_exp = abcd_u ** int(self.N_u / 2)
        abcd_sc = u_exp @ (abcd_l**self.N_l) @ u_exp
        return u_exp @ (abcd_sc**self.N_sc) @ u_exp
