"""Utility function for mathematic expressions."""
import numba as nb
import numpy as np


@nb.njit("(float64[:,:,:], float64[:,:,:])")
def matmul_2x2(matrices_a: np.ndarray, matrices_b: np.ndarray) -> np.ndarray:
    """Fast multiplication between list of 2x2 matrices."""
    assert matrices_a.shape == matrices_b.shape
    assert matrices_a.shape[1] == 2 and matrices_a.shape[2] == 2

    n_mat = matrices_a.shape[0]
    result_matrices = np.empty((n_mat, 2, 2))
    for k in range(n_mat):
        for i in range(2):
            for j in range(2):
                result_matrices[k, i, j] = (
                    matrices_a[k, i, 0] * matrices_b[k, 0, j]
                    + matrices_a[k, i, 1] * matrices_b[k, 1, j]
                )

    return result_matrices


@nb.njit("(float64[:,:,:], int32)")
def matpow_2x2(matrices_a: np.ndarray, exponent: int) -> np.ndarray:
    """Fast exponentiation of list of 2x2 matrices."""
    assert matrices_a.shape[1] == 2 and matrices_a.shape[2] == 2
    assert exponent > 1
    n_mat = matrices_a.shape[0]
    result_matrices = matrices_a.copy()
    partial_exp = np.empty((2, 2))
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
