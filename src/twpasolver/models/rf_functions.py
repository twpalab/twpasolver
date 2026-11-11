"""Compiled functions for components impedance and abcd matrices."""

import numpy as np
from numba import njit

from twpasolver.bonus_types import ComplexArray, FloatArray


@njit(cache=True)
def inductance(freqs: FloatArray, L: float) -> ComplexArray:
    """
    Impedance ``2j pi f L`` of an inductance.

    Args:
        freqs (FloatArray): Frequencies in Hz.
        L (float): Inductance in H.
    """
    assert L >= 0
    return (2j * np.pi * freqs * L).astype(np.complex128)


@njit(cache=True)
def capacitance(freqs: FloatArray, C: float) -> ComplexArray:
    """
    Impedance ``-1j / (2 pi f C)`` of a capacitance.

    Args:
        freqs (FloatArray): Frequencies in Hz.
        C (float): Capacitance in F.
    """
    assert np.real(C) >= 0
    return (-1j / (2 * np.pi * freqs * C)).astype(np.complex128)


@njit(cache=True)
def stub(
    freqs: FloatArray, L: float, C: complex, length: float, open: bool = True
) -> ComplexArray:
    """
    Impedance of an open or shorted stub.

    Args:
        freqs (FloatArray): Frequencies in Hz.
        L (float): Inductance per unit length.
        C (float): Capacitance per unit length.
        length (float): Length of the stub.
        open (bool): Open stub if True, shorted stub otherwise.
    """
    beta_l = 2 * np.pi * freqs * np.sqrt(L * C) * length
    Z = np.sqrt(L / C)
    if open:
        return -1j * Z / np.tan(beta_l)
    return 1j * Z * np.tan(beta_l)


@njit(cache=True)
def parallel_admittance_abcd(Y: ComplexArray) -> ComplexArray:
    """
    ABCD matrices ``[[1, 0], [Y, 1]]`` of a shunt admittance, shape (len(Y), 2, 2).
    """
    abcd = np.zeros((len(Y), 2, 2), dtype=np.complex128)
    abcd[:, 0, 0] = 1
    abcd[:, 1, 1] = 1
    abcd[:, 1, 0] = Y
    return abcd


@njit(cache=True)
def series_impedance_abcd(Z: ComplexArray) -> ComplexArray:
    """
    ABCD matrices ``[[1, Z], [0, 1]]`` of a series impedance, shape (len(Z), 2, 2).
    """
    abcd = np.zeros((len(Z), 2, 2), dtype=np.complex128)
    abcd[:, 0, 0] = 1
    abcd[:, 1, 1] = 1
    abcd[:, 0, 1] = Z
    return abcd


@njit(cache=True)
def lossless_line_abcd(freqs: FloatArray, C: float, L: float, l: float) -> ComplexArray:
    """
    ABCD matrices of a lossless transmission line.

    Args:
        freqs (FloatArray): Frequencies in Hz.
        C (float): Capacitance per unit length.
        L (float): Inductance per unit length.
        l (float): Length of the transmission line.
    """
    assert np.real(C) >= 0
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


@njit(cache=True)
def LCLf_abcd(
    freqs: FloatArray, C: complex, L: float, Lf: float, qp_loss: float = 0.0
) -> ComplexArray:
    """
    ABCD matrices of an LCLf cell.

    Args:
        freqs (FloatArray): Frequencies in Hz.
        C (complex): Capacitance (complex for dielectric loss).
        L (float): Series inductance.
        Lf (float): Finger inductance, in series with ``C``.
        qp_loss (float): Quasiparticle-loss coefficient (units 1/Hz).
    """
    assert np.real(C) >= 0
    assert L >= 0
    assert Lf >= 0
    n_mat = len(freqs)
    abcd = np.empty((n_mat, 2, 2), dtype=np.complex128)
    for i in range(n_mat):
        w = 2 * np.pi * freqs[i]
        L_eff = L * (1 - 1j * qp_loss * freqs[i])
        den = 2 - Lf * C * w**2
        abcd[i, 0, 0] = 1
        abcd[i, 0, 1] = 1j * L_eff * w
        abcd[i, 1, 0] = 1j * 2 * C * w / den
        abcd[i, 1, 1] = 1 - 2 * L_eff * C * w**2 / den
    return abcd


@njit(cache=True)
def get_stub_cell(
    freqs: FloatArray, C: complex, L: float, l1: float, l2: float
) -> ComplexArray:
    """
    ABCD matrices of a stub cell.

    Args:
        freqs (FloatArray): Frequencies in Hz.
        C (float): Capacitance per unit length.
        L (float): Inductance per unit length.
        l1 (float): Length of the first section of the stub.
        l2 (float): Length of the second section of the stub.
    """
    assert np.real(C) >= 0
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
