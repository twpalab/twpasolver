"""
Numba-compiled utilities for arrays of 2x2 matrices.

Products, powers, ABCD/S/Z conversions and dB/dBm unit conversions.
"""

import numba as nb
import numpy as np

from twpasolver.bonus_types import ComplexArray, FloatArray


@nb.njit(cache=True)
def matmul_2x2(
    matrices_a: ComplexArray,
    matrices_b: ComplexArray,
) -> ComplexArray:
    """Product of two arrays of 2x2 matrices, shape (n, 2, 2)."""
    assert matrices_a.shape == matrices_b.shape
    assert matrices_a.shape[1] == 2 and matrices_a.shape[2] == 2

    n_mat = matrices_a.shape[0]
    result_matrices = np.empty((n_mat, 2, 2), dtype=np.complex128)
    for k in range(n_mat):
        for i in range(2):
            for j in range(2):
                result_matrices[k, i, j] = (
                    matrices_a[k, i, 0] * matrices_b[k, 0, j]
                    + matrices_a[k, i, 1] * matrices_b[k, 1, j]
                )

    return result_matrices


@nb.njit(cache=True)
def matpow_2x2(matrices_a: ComplexArray, exponent: int) -> ComplexArray:
    """Positive integer power of an array of 2x2 matrices, by repeated squaring."""
    assert matrices_a.shape[1] == 2 and matrices_a.shape[2] == 2
    assert exponent > 0
    base = matrices_a.astype(np.complex128)
    result = base.copy()
    have_result = False
    e = exponent
    while e > 0:
        if e & 1:
            if have_result:
                result = matmul_2x2(result, base)
            else:
                result = base.copy()
                have_result = True
        e >>= 1
        if e > 0:
            base = matmul_2x2(base, base)
    return result


@nb.njit(cache=True)
def a2s(abcd: ComplexArray, Z0: complex | float) -> ComplexArray:
    """
    Convert arrays of ABCD matrices to S-parameters.

    Args:
        abcd (ComplexArray): ABCD matrices, shape (n, 2, 2).
        Z0 (complex | float): Reference impedance in Ohm (positive real part).
    """
    assert abcd.shape[1] == 2 and abcd.shape[2] == 2
    assert np.real(Z0) > 0
    n_mat = abcd.shape[0]
    spar_mat = np.empty((n_mat, 2, 2), dtype=np.complex128)
    for i in range(n_mat):
        A = abcd[i, 0, 0]
        BZ = abcd[i, 0, 1] / Z0
        CZ = abcd[i, 1, 0] * Z0
        D = abcd[i, 1, 1]
        norm = A + BZ + CZ + D
        spar_mat[i, 0, 0] = (A + BZ - CZ - D) / norm
        spar_mat[i, 0, 1] = 2.0 * (A * D - BZ * CZ) / norm
        spar_mat[i, 1, 0] = 2.0 / norm
        spar_mat[i, 1, 1] = (-A + BZ - CZ + D) / norm
    return spar_mat


@nb.njit(cache=True)
def s2a(spar: ComplexArray, Z0: complex | float) -> ComplexArray:
    """
    Convert arrays of S-parameters to ABCD matrices.

    Args:
        spar (ComplexArray): S-parameter matrices, shape (n, 2, 2).
        Z0 (complex | float): Reference impedance in Ohm (positive real part).
    """
    assert spar.shape[1] == 2 and spar.shape[2] == 2
    assert np.real(Z0) > 0
    n_mat = spar.shape[0]
    abcd = np.empty((n_mat, 2, 2), dtype=np.complex128)
    for i in range(n_mat):
        S11 = spar[i, 0, 0]
        S12 = spar[i, 0, 1]
        S21 = spar[i, 1, 0]
        S22 = spar[i, 1, 1]
        abcd[i, 0, 0] = ((1 + S11) * (1 - S22) + S12 * S21) / (2 * S21)
        abcd[i, 0, 1] = ((1 + S11) * (1 + S22) - S12 * S21) / (2 * S21) * Z0
        abcd[i, 1, 0] = ((1 - S11) * (1 - S22) - S12 * S21) / (2 * S21) * 1 / Z0
        abcd[i, 1, 1] = ((1 - S11) * (1 + S22) + S12 * S21) / (2 * S21)
    return abcd


@nb.njit(cache=True)
def a2z(abcd: ComplexArray) -> ComplexArray:
    """Convert arrays of ABCD matrices, shape (n, 2, 2), to Z-parameters."""
    assert abcd.shape[1] == 2 and abcd.shape[2] == 2
    n_mat = abcd.shape[0]
    z_mat = np.empty((n_mat, 2, 2), dtype=np.complex128)
    for i in range(n_mat):
        A = abcd[i, 0, 0]
        B = abcd[i, 0, 1]
        C = abcd[i, 1, 0]
        D = abcd[i, 1, 1]
        z_mat[i, 0, 0] = A / C
        z_mat[i, 0, 1] = (A * D - B * C) / C
        z_mat[i, 1, 0] = 1 / C
        z_mat[i, 1, 1] = D / C
    return z_mat


@nb.njit(cache=True)
def z2a(z_mat: ComplexArray) -> ComplexArray:
    """Convert arrays of Z-parameters, shape (n, 2, 2), to ABCD matrices."""
    assert z_mat.shape[1] == 2 and z_mat.shape[2] == 2
    n_mat = z_mat.shape[0]
    abcd = np.empty((n_mat, 2, 2), dtype=np.complex128)
    for i in range(n_mat):
        Z11 = z_mat[i, 0, 0]
        Z12 = z_mat[i, 0, 1]
        Z21 = z_mat[i, 1, 0]
        Z22 = z_mat[i, 1, 1]
        abcd[i, 0, 0] = Z11 / Z21
        abcd[i, 0, 1] = (Z11 * Z22 - Z12 * Z21) / Z21
        abcd[i, 1, 0] = 1 / Z21
        abcd[i, 1, 1] = Z22 / Z21
    return abcd


@nb.njit(cache=True)
def to_dB(values: FloatArray | ComplexArray) -> FloatArray:
    """Convert amplitudes to dB, ``20 log10 |values|``."""
    return 20.0 * np.log10(np.abs(values))


@nb.njit(cache=True)
def dBm_to_I(power: float, Z0: float = 50) -> float:
    """
    Convert power in dBm to current amplitude in A, with ``P = Z0 I^2 / 2``.

    Args:
        power (float): Power in dBm.
        Z0 (float, optional): Reference impedance in Ohm. Defaults to 50.
    """
    pw = 10 ** (power / 10) / 1000  # power in W
    return np.sqrt(pw / Z0 * 2)  # current amplitude, in A


@nb.njit(cache=True)
def I_to_dBm(curr: float, Z0: float = 50) -> float:
    """
    Convert current amplitude in A to power in dBm, with ``P = Z0 I^2 / 2``.

    Args:
        curr (float): Current amplitude in A.
        Z0 (float, optional): Reference impedance in Ohm. Defaults to 50.
    """
    pw = curr**2 / 2 * Z0
    return 10 * np.log10(np.abs(pw) * 1000)
