"""Compiled functions for components impedance and abcd matrices."""

import numpy as np
from numba import njit

from twpasolver.bonus_types import ComplexArray, FloatArray


@njit
def inductance(freqs: FloatArray, L: float) -> ComplexArray:
    """
    Calculate the impedance of an inductance as a function of frequency.

    Args:
        freqs (FloatArray): Array of frequencies.
        L (float): Inductance value.

    Returns:
        np.ndarray: Impedance of the inductance.
    """
    assert L >= 0
    return 2j * np.pi * freqs * L


@njit
def capacitance(freqs: FloatArray, C: float) -> ComplexArray:
    """
    Calculate the impedance of a capacitor as a function of frequency.

    Args:
        freqs (FloatArray): Array of frequencies.
        C (float): Capacitance value.

    Returns:
        np.ndarray: Impedance of the capacitor.
    """
    assert C >= 0
    return -1j / (2 * np.pi * freqs * C)


@njit
def stub(
    freqs: FloatArray, L: float, C: float, length: float, open: bool = True
) -> ComplexArray:
    """
    Calculate the impedance of a stub as a function of frequency.

    Args:
        freqs (FloatArray): Array of frequencies.
        L (float): Inductance value.
        C (float): Capacitance value.
        length (float): Length of the stub.
        open (bool): If True, calculate for an open stub; otherwise, for a short stub.

    Returns:
        np.ndarray: Impedance of the stub.
    """
    beta_l = 2 * np.pi * freqs * np.sqrt(L * C) * length
    Z = np.sqrt(L / C)
    if open:
        return -1j * Z / np.tan(beta_l)
    return 1j * Z * np.tan(beta_l)


@njit
def parallel_admittance_abcd(Y: ComplexArray) -> ComplexArray:
    """
    Calculate the ABCD matrix of a parallel admittance.

    Args:
        Y (ComplexArray): Array of admittance values.

    Returns:
        np.ndarray: ABCD matrix of the parallel admittance.
    """
    abcd = np.zeros((len(Y), 2, 2), dtype=np.complex128)
    abcd[:, 0, 0] = 1
    abcd[:, 1, 1] = 1
    abcd[:, 1, 0] = Y
    return abcd


@njit
def series_impedance_abcd(Z: ComplexArray) -> ComplexArray:
    """
    Calculate the ABCD matrix of a series impedance.

    Args:
        Z (ComplexArray): Array of impedance values.

    Returns:
        np.ndarray: ABCD matrix of the series impedance.
    """
    abcd = np.zeros((len(Z), 2, 2), dtype=np.complex128)
    abcd[:, 0, 0] = 1
    abcd[:, 1, 1] = 1
    abcd[:, 0, 1] = Z
    return abcd


@njit
def lossless_line_abcd(freqs: FloatArray, C: float, L: float, l: float) -> ComplexArray:
    """
    Calculate the ABCD matrix of a lossless transmission line.

    Args:
        freqs (FloatArray): Array of frequencies.
        C (float): Capacitance per unit length.
        L (float): Inductance per unit length.
        l (float): Length of the transmission line.

    Returns:
        np.ndarray: ABCD matrix of the lossless transmission line.
    """
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
def LCLf_abcd(freqs: FloatArray, C: float, L: float, Lf: float) -> ComplexArray:
    """
    Calculate the ABCD matrix of an LCLf cell model.

    Args:
        freqs (FloatArray): Array of frequencies.
        C (float): Capacitance value.
        L (float): Inductance value.
        Lf (float): Additional inductance value.

    Returns:
        np.ndarray: ABCD matrix of the LCLf cell model.
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
    freqs: FloatArray, C: float, L: float, l1: float, l2: float
) -> ComplexArray:
    """
    Calculate the ABCD matrix of a stub cell model.

    Args:
        freqs (FloatArray): Array of frequencies.
        C (float): Capacitance value.
        L (float): Inductance value.
        l1 (float): Length of the first section of the stub.
        l2 (float): Length of the second section of the stub.

    Returns:
        np.ndarray: ABCD matrix of the stub cell model.
    """
    assert C >= 0
    assert L >= 0
    Z0 = np.sqrt(L / C)
    v = np.sqrt(L * C) * l1
    n_mat = len(freqs)
    abcd = np.empty((n_mat, 2, 2), dtype=np.complex128)
    for i in range(n_mat):
        w = 2 * np.pi * freqs[i]
        beta = w * v
        Z2 = 1j * L * l2 * w
        Z3_inv = 2 * 1j * np.tan(beta) / Z0

        abcd[i, 0, 0] = 1
        abcd[i, 0, 1] = Z2
        abcd[i, 1, 0] = Z3_inv
        abcd[i, 1, 1] = 1 + Z2 * Z3_inv
    return abcd
