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


@njit
def get_stub_cell(
    freqs: np.ndarray[Any, np.dtype[np.float64]],
    C: float,
    L: float,
    Lf: float,
) -> complex_array:
    """Get base abcd matrix of stub cell model."""
    assert C >= 0
    assert L >= 0
    assert Lf >= 0
    Z0 = np.sqrt(Lf / C)
    n_mat = len(freqs)
    abcd = np.empty((n_mat, 2, 2), dtype=np.complex128)
    for i in range(n_mat):
        w = 2 * np.pi * freqs[i]
        beta = w * np.sqrt(Lf * C)

        abcd[i, 0, 0] = 1

        abcd[i, 0, 1] = 1j * L * w

        abcd[i, 1, 0] = 0.5 * 1j * np.tan(beta) / Z0
        abcd[i, 1, 1] = 1 - 0.5 * L * w * np.tan(beta) / Z0

    return abcd


@njit
def parallel_admittance_abcd(Y: complex_array):
    """Get abcd matrix of parallel admittance."""
    abcd = np.zeros((len(Y), 2, 2), dtype=np.complex128)
    abcd[:, 0, 0] = 1
    abcd[:, 1, 1] = 1
    abcd[:, 1, 0] = Y
    return abcd


@njit
def series_impedance_abcd(Z: complex_array):
    """Get abcd matrix of series impedance."""
    abcd = np.zeros((len(Z), 2, 2), dtype=np.complex128)
    abcd[:, 0, 0] = 1
    abcd[:, 1, 1] = 1
    abcd[:, 0, 1] = Z
    return abcd


def lossless_line(
    freqs: np.ndarray[Any, np.dtype[np.float64]],
    C: float,
    L: float,
) -> complex_array:
    """Get base abcd matrix of lossless line."""
    assert C >= 0
    assert L >= 0
    Z0 = np.sqrt(L / C)
    n_mat = len(freqs)
    abcd = np.empty((n_mat, 2, 2), dtype=np.complex128)
    for i in range(n_mat):
        w = 2 * np.pi * freqs[i]
        beta = w * np.sqrt(L * C)
        abcd[i, 0, 0] = np.cos(beta)

        abcd[i, 0, 1] = 1j * Z0 * np.sin(beta)
        abcd[i, 1, 0] = 1j * np.sin(beta) / Z0
        abcd[i, 1, 1] = np.cos(beta)

    return abcd


class StubBaseCell(TwoPortModel):
    """Base cell of Malnou model."""

    C: NonNegativeFloat
    L: NonNegativeFloat
    Lf: NonNegativeFloat
    l1: NonNegativeFloat
    l2: NonNegativeFloat
    n_stub: int = 2

    def get_abcd(self, freqs: np.ndarray) -> ABCDArray:
        """Compute abcd matrix."""
        w = freqs * 2 * np.pi
        stub_Y = (
            self.n_stub
            * 1j
            * np.tan(w * np.sqrt(self.C * self.Lf))
            / np.sqrt(self.Lf / self.C)
        )
        stub = ABCDArray(parallel_admittance_abcd(stub_Y))
        line = ABCDArray(series_impedance_abcd(1j * w * self.L))
        return line @ stub


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

    unloaded: MalnouBaseCell | StubBaseCell
    loaded: MalnouBaseCell | StubBaseCell
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
