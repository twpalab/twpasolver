"""Utility function for mathematic expressions."""
from typing import Any, TypeAlias

import numba as nb
import numpy as np

complex_array: TypeAlias = np.ndarray[Any, np.dtype[np.complex128]]


@nb.njit
def matmul_2x2(
    matrices_a: complex_array,
    matrices_b: complex_array,
) -> complex_array:
    """Fast multiplication between list of 2x2 matrices."""
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


@nb.njit
def matpow_2x2(matrices_a: complex_array, exponent: int) -> complex_array:
    """Fast exponentiation of list of 2x2 matrices with recursion."""
    assert matrices_a.shape[1] == 2 and matrices_a.shape[2] == 2
    assert exponent > 0
    if exponent == 1:
        return matrices_a.astype(np.complex128)
    elif exponent % 2 == 0:
        result_matrices = matpow_2x2(matrices_a, int(exponent / 2))
        return matmul_2x2(result_matrices, result_matrices)
    result_matrices = matrices_a.astype(np.complex128)
    partial_exp = np.empty((2, 2), dtype=np.complex128)
    n_mat = matrices_a.shape[0]
    for k in range(n_mat):
        for _ in range(1, exponent):
            for i in range(2):
                for j in range(2):
                    partial_exp[i, j] = (
                        matrices_a[k, i, 0] * result_matrices[k, 0, j]
                        + matrices_a[k, i, 1] * result_matrices[k, 1, j]
                    )
            result_matrices[k, :, :] = partial_exp
    return result_matrices


@nb.njit
def a2s(abcd: complex_array, Z0: float) -> complex_array:
    """Convert list of ABCD matrices to list of S parameters."""
    assert abcd.shape[1] == 2 and abcd.shape[2] == 2
    assert Z0 > 0
    n_mat = abcd.shape[0]
    spar_mat = np.empty((n_mat, 2, 2), dtype=np.complex128)
    for i in range(n_mat):
        A = abcd[i, 0, 0]
        B = abcd[i, 0, 1]
        C = abcd[i, 1, 0]
        D = abcd[i, 1, 1]
        spar_mat[i, 0, 0] = (A + B / Z0 - C * Z0 - D) / (A + B / Z0 + C * Z0 + D)
        spar_mat[i, 0, 1] = 2.0 * (A * D - B * C) / (A + B / Z0 + C * Z0 + D)
        spar_mat[i, 1, 0] = 2.0 / (A + B / Z0 + C * Z0 + D)
        spar_mat[i, 1, 1] = (-A + B / Z0 - C * Z0 + D) / (A + B / Z0 + C * Z0 + D)
    return spar_mat


@nb.njit
def s2a(spar: complex_array, Z0: float) -> complex_array:
    """Convert list of S parameters to list of ABCD matrices."""
    assert spar.shape[1] == 2 and spar.shape[2] == 2
    assert Z0 > 0
    n_mat = spar.shape[0]
    abcd = np.empty((n_mat, 2, 2), dtype=np.complex128)
    for i in range(n_mat):
        S11 = spar[i, 0, 0]
        S12 = spar[i, 0, 1]
        S21 = spar[i, 1, 0]
        S22 = spar[i, 1, 1]
        abcd[i, 0, 0] = ((1 + S11) * (1 - S22) + S12 * S21) / (2 * S21)
        abcd[i, 0, 1] = ((1 + S11) * (1 + S22) - S12 * S21) / (2 * S21) * Z0
        abcd[i, 1, 0] = ((1 - S11) * (1 - S22) - S12 * S21) / (2 * S21) * 1 / 0
        abcd[i, 1, 1] = ((1 - S11) * (1 + S22) + S12 * S21) / (2 * S21)
    return abcd
