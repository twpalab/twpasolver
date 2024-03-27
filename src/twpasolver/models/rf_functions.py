"""Compiled functions for components impedance and abcd matrices."""

import numpy as np
from numba import njit

from twpasolver.typing import complex_array, float_array


@njit
def inductance(freqs: float_array, L: float):
    """Impedance of inductance as function of frequency."""
    assert L >= 0
    return 2j * np.pi * freqs * L


@njit
def capacitance(freqs: float_array, C: float):
    """Impedance of capacitor as function of frequency."""
    assert C >= 0
    return -1j / (2 * np.pi * freqs * C)


@njit
def stub(freqs: float_array, L: float, C: float, length: float, open: bool = True):
    """Impedance of stub as function of frequency."""
    beta_l = 2 * np.pi * freqs * np.sqrt(L * C) * length
    Z = np.sqrt(L / C)
    if open:
        return -1j * Z / np.tan(beta_l)
    return 1j * Z * np.tan(beta_l)


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


@njit
def lossless_line_abcd(
    freqs: float_array, C: float, L: float, l: float
) -> complex_array:
    """Get base abcd matrix of lossless line."""
    assert C >= 0
    assert L >= 0
    Z0 = np.sqrt(L / C)
    n_mat = len(freqs)
    abcd = np.empty((n_mat, 2, 2), dtype=np.complex128)
    for i in range(n_mat):
        beta = 2 * np.pi * freqs[i] * np.sqrt(L * C) * l
        abcd[i, 0, 0] = np.cos(beta)

        abcd[i, 0, 1] = 1j * Z0 * np.sin(beta)
        abcd[i, 1, 0] = 1j * np.sin(beta) / Z0
        abcd[i, 1, 1] = np.cos(beta)

    return abcd


@njit
def LCLf_abcd(freqs: float_array, C: float, L: float, Lf: float) -> complex_array:
    """
    Get abcd matrix of LCLf cell model.

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
    freqs: float_array,
    C: float,
    L: float,
    Lf: float,
) -> complex_array:
    """Get abcd matrix of stub cell model."""
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
