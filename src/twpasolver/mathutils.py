"""
Utility functions for mathematical expressions and RF equations.

This module provides a set of utility functions optimized with Numba for high performance.
These functions include matrix multiplications, conversions between different parameter
representations, and solutions for coupled mode equations used in RF engineering.
"""

from typing import Tuple

import numba as nb
import numpy as np
from CyRK import nbrk_ode

from twpasolver.typing import ComplexArray, FloatArray

nb_complex3d = nb.complex128[:, :, :]
nb_complex1d = nb.complex128[:]
nb_float1d = nb.float64[:]


@nb.njit(cache=True)
def matmul_2x2(
    matrices_a: ComplexArray,
    matrices_b: ComplexArray,
) -> ComplexArray:
    """
    Fast multiplication between arrays of 2x2 matrices.

    Args:
        matrices_a (ComplexArray): Array of 2x2 complex matrices.
        matrices_b (ComplexArray): Array of 2x2 complex matrices.

    Returns:
        ComplexArray: Resultant array of 2x2 complex matrices after multiplication.
    """
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
    """
    Fast exponentiation of arrays of 2x2 matrices using recursion.

    Args:
        matrices_a (ComplexArray): Array of 2x2 complex matrices.
        exponent (int): Exponent to which matrices are to be raised.

    Returns:
        ComplexArray: Resultant array of 2x2 complex matrices after exponentiation.
    """
    assert matrices_a.shape[1] == 2 and matrices_a.shape[2] == 2
    assert exponent > 0
    if exponent == 1:
        return matrices_a.astype(np.complex128)
    if exponent % 2 == 0:
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


@nb.njit(cache=True)
def a2s(abcd: ComplexArray, Z0: complex | float) -> ComplexArray:
    """
    Convert arrays of ABCD matrices to arrays of S-parameters.

    Args:
        abcd (ComplexArray): Array of 2x2 ABCD matrices.
        Z0 (complex | float): Reference impedance.

    Returns:
        ComplexArray: Array of 2x2 S-parameter matrices.
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
    Convert arrays of S-parameters to arrays of ABCD matrices.

    Args:
        spar (ComplexArray): Array of 2x2 S-parameter matrices.
        Z0 (complex | float): Reference impedance.

    Returns:
        ComplexArray: Array of 2x2 ABCD matrices.
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
def to_dB(values: FloatArray | ComplexArray) -> FloatArray:
    """
    Convert arrays of values to dB.

    Args:
        values (FloatArray | ComplexArray): Array of values to be converted to dB.

    Returns:
        FloatArray: Array of values in dB.
    """
    return 20.0 * np.log10(np.abs(values))


@nb.njit(cache=True)
def dBm_to_I(power: float, Z0: float = 50) -> float:
    """
    Convert power from dBm to current in amperes.

    Args:
        power (float): Power in dBm.
        Z0 (float, optional): Reference impedance, default is 50 ohms.

    Returns:
        float: Current amplitude in amperes.
    """
    pw = 10 ** (power / 10) / 1000  # power in W
    return np.sqrt(pw / Z0 * 2)  # current amplitude, in A


@nb.njit(cache=True)
def I_to_dBm(curr: float, Z0: float = 50) -> float:
    """
    Convert current in amperes to power in dBm.

    Args:
        curr (float): Current amplitude in amperes.
        Z0 (float, optional): Reference impedance, default is 50 ohms.

    Returns:
        float: Power in dBm.
    """
    pw = curr**2 / 2 * Z0
    return 10 * np.log10(pw * 1000)


@nb.njit(cache=True)
def compute_phase_matching(
    freqs: FloatArray,
    pump_freqs: FloatArray,
    k_signal_array: FloatArray,
    k_pump_array: FloatArray,
    chi: float,
) -> Tuple[FloatArray, FloatArray, FloatArray]:
    """
    Compute phase matching profiles and triplets for given frequencies.

    Args:
        freqs (FloatArray): Array of signal frequencies.
        pump_freqs (FloatArray): Array of pump frequencies.
        k_signal_array (FloatArray): Array of signal wave numbers.
        k_pump_array (FloatArray): Array of pump wave numbers.
        chi (float): Nonlinear coefficient.

    Returns:
        Tuple[FloatArray, FloatArray, FloatArray]: Phase matching profile, frequency triplets, and wave number triplets.
    """
    num_freqs = len(freqs)
    num_pumps = len(pump_freqs)

    delta_k = np.empty(shape=(num_freqs, num_pumps))
    freq_triplets = np.empty((num_pumps, 3))
    k_triplets = np.empty((num_pumps, 3))

    for i in range(num_pumps):
        p_f = pump_freqs[i]
        k_pump = k_pump_array[i]
        k_idler = np.interp(p_f - freqs, freqs, k_signal_array)

        deltas = (
            k_pump
            - k_signal_array
            - k_idler
            + chi * (k_pump - 2 * k_idler - 2 * k_signal_array)
        )
        delta_k[:, i] = deltas

        half_pump_idx = np.searchsorted(freqs, p_f / 2)

        max_delta_idx = np.argmax(deltas[:half_pump_idx])
        min_k_idx = (
            np.argmin(np.abs(deltas[max_delta_idx:half_pump_idx])) + max_delta_idx
        )

        freq_triplets[i] = np.array([p_f, freqs[min_k_idx], p_f - freqs[min_k_idx]])
        k_triplets[i] = np.array(
            [k_pump, k_signal_array[min_k_idx], k_idler[min_k_idx]]
        )

    return delta_k, freq_triplets, k_triplets


@nb.njit(
    nb_complex1d(
        nb.float32,
        nb_complex1d,
        nb.float32,
        nb.float32,
        nb.float32,
        nb.float32,
        nb.float32,
    ),
    cache=True,
)
def CMEode_complete(
    t: float,
    y: ComplexArray,
    kp: float,
    ks: float,
    ki: float,
    xi: float,
    epsi: float,
) -> ComplexArray:
    """
    Complete coupled mode equation model for current amplitudes.

    y[0] = Ip # pump current
    y[1] = Is # signal current
    y[2] = Ii # idler current
    t    = x  #

    Equations A5a, A5b and A5c from
    `PRX Quantum 2, 010302 (2021) <https://doi.org/10.1103/PRXQuantum.2.010302>`_

    Args:
        t (float): Time variable.
        y (ComplexArray): Array of current amplitudes [Ip, Is, Ii].
        kp (float): Pump wave number.
        ks (float): Signal wave number.
        ki (float): Idler wave number.
        xi (float): Nonlinear coefficient.
        epsi (float): Small perturbation parameter.

    Returns:
        ComplexArray: Derivatives of current amplitudes [dIp/dt, dIs/dt, dIi/dt].
    """
    dk = kp - ks - ki

    a = 1j * epsi / 4
    b = 1j * xi / 8

    # Compute the magnitudes of currents
    abs_y0_sq = abs(y[0]) ** 2
    abs_y1_sq = abs(y[1]) ** 2
    abs_y2_sq = abs(y[2]) ** 2

    # Compute intermediate values
    pp = abs_y0_sq + 2 * abs_y1_sq + 2 * abs_y2_sq
    ss = 2 * abs_y0_sq + abs_y1_sq + 2 * abs_y2_sq
    ii = 2 * abs_y0_sq + 2 * abs_y1_sq + abs_y2_sq

    # Compute derivatives
    derivs = np.empty(3, dtype=np.complex128)
    derivs[0] = a * kp * y[1] * y[2] * np.exp(-1j * dk * t) + b * kp * y[0] * pp
    derivs[1] = a * ks * y[0] * np.conj(y[2]) * np.exp(1j * dk * t) + b * ks * y[1] * ss
    derivs[2] = a * ki * y[0] * np.conj(y[1]) * np.exp(1j * dk * t) + b * ki * y[2] * ii

    return derivs


def cme_solve(
    k_signal: FloatArray,
    k_idler: FloatArray,
    x_array: FloatArray,
    y0: ComplexArray,
    k_pump: float,
    xi: float,
    epsi: float,
) -> ComplexArray:
    """
    Solve coupled mode equations for multiple frequencies.

    Args:
        k_signal (FloatArray): Array of signal wave numbers.
        k_idler (FloatArray): Array of idler wave numbers.
        x_array (FloatArray): Array of position values for evaluation.
        y0 (ComplexArray): Initial conditions for the coupled mode equations.
        k_pump (float): Pump wave number.
        xi (float): Nonlinear coefficient.
        epsi (float): Small perturbation parameter.

    Returns:
        ComplexArray: Solution of the coupled mode equations for the given frequencies.
    """
    len_k = len(k_signal)

    x_span = (x_array[0], x_array[-1])
    I_triplets = np.empty((len_k, 3, len(x_array)), dtype=np.complex128)
    for i in nb.prange(len_k):
        k = k_signal[i]
        _, I_triplets[i], _, _ = nbrk_ode(
            CMEode_complete,
            x_span,
            y0,
            args=(
                k_pump,
                k,
                k_idler[i],
                xi,
                epsi,
            ),
            atol=1e-16,
            rtol=1e-10,
            max_num_steps=1000,
            first_step=1,
            t_eval=x_array,
            rk_method=1,
        )
    return I_triplets
