"""
Utility functions for mathematical expressions and RF equations.

This module provides a set of utility functions optimized with Numba for high performance.
These functions include matrix multiplications, conversions between different parameter
representations, and solutions for coupled mode equations used in RF engineering.
"""

import numba as nb
import numpy as np
from CyRK import nbrk_ode

from twpasolver.bonus_types import (
    ComplexArray,
    FloatArray,
    nb_complex1d,
    nb_float1d,
    nb_int2d,
)


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


@nb.njit(
    nb_complex1d(
        nb.float64,
        nb_complex1d,
        nb_float1d,
        nb_int2d,
        nb_int2d,
        nb_float1d,
        nb_float1d,
    ),
    cache=True,
)
def general_cme_no_reflections(
    x,
    currents,
    kappas,
    relations_3wm,
    relations_4wm,
    coeffs_3wm,
    coeffs_4wm,
):
    """Return derivatives of Coupled Mode Equations system including arbitrary 3WM and 4WM relations."""
    num_modes = len(currents)
    exp_pos = np.exp(1j * kappas * x)
    alphas_rhs = np.empty(2 * num_modes, dtype=np.complex128)
    for i in range(num_modes):
        alphas_rhs[i] = currents[i] * exp_pos[i]
        alphas_rhs[i + num_modes] = np.conj(alphas_rhs[i])
    derivs = np.zeros(num_modes, dtype=np.complex128)

    for idx, idx1, idx2 in relations_3wm:
        derivs[idx] += coeffs_3wm[idx] * alphas_rhs[idx1] * alphas_rhs[idx2]
    for idx, idx1, idx2, idx3 in relations_4wm:
        derivs[idx] += (
            coeffs_4wm[idx] * alphas_rhs[idx1] * alphas_rhs[idx2] * alphas_rhs[idx3]
        )
    # return derivs
    return 1j * derivs * np.conj(exp_pos) * kappas


@nb.njit(
    nb_complex1d(
        nb.float64,
        nb_complex1d,
        nb_float1d,
        nb_complex1d,
        nb_complex1d,
        nb_int2d,
        nb_int2d,
        nb_float1d,
        nb_float1d,
    ),
    cache=True,
)
def general_cme(
    x,
    currents,
    kappas,
    ts_reflection,
    ts_reflection_neg,
    relations_3wm,
    relations_4wm,
    coeffs_3wm,
    coeffs_4wm,
):
    """Return derivatives of Coupled Mode Equations system including reflection and arbitrary 3WM and 4WM relations."""
    num_modes = len(currents)
    common_term_pos = ts_reflection * np.exp(1j * kappas * x)
    common_term_neg = 0.0 * np.exp(-1j * kappas * x)
    alphas_rhs = np.empty(2 * num_modes, dtype=np.complex128)
    for i in range(num_modes):
        alphas_rhs[i] = currents[i] * (common_term_pos[i] + common_term_neg[i])
        alphas_rhs[i + num_modes] = np.conj(alphas_rhs[i])
    derivs = np.zeros(num_modes, dtype=np.complex128)

    for idx, idx1, idx2 in relations_3wm:
        derivs[idx] += coeffs_3wm[idx] * alphas_rhs[idx1] * alphas_rhs[idx2]
    for idx, idx1, idx2, idx3 in relations_4wm:
        derivs[idx] += (
            coeffs_4wm[idx] * alphas_rhs[idx1] * alphas_rhs[idx2] * alphas_rhs[idx3]
        )
    return 1j * derivs * kappas / (common_term_pos - common_term_neg)


def cme_general_solve(
    x_array: FloatArray,
    y0: ComplexArray,
    data_kappas_gammas,
    relations_coefficients,
    thin: int = 200,
    reflections: bool = True,
) -> ComplexArray:
    """
    Solve coupled mode equations for multiple frequencies.

    Args:
        x_array (FloatArray): Array of position values for evaluation.
        y0 (ComplexArray): Initial currents for each mode.
        data_kappas_gammas: Kappas and reflection coefficients for all modes.
        relations_coefficients: tables with indexes representing mixing relations between modes and relative coefficients.
        thin (int): Thinning factor for plotting.

    Returns:
        ComplexArray: Solution of the coupled mode equations for the given frequencies.
    """
    if reflections:
        cme_model = general_cme
    else:
        cme_model = general_cme_no_reflections

    n_modes = len(y0)
    relations_coefficients = [np.array(rel) for rel in relations_coefficients]
    x_span = (x_array[0], x_array[-1])
    I_triplets = np.empty((n_modes, len(x_array)), dtype=np.complex128)

    _, I_triplets, _, _ = nbrk_ode(
        cme_model,
        x_span,
        y0,
        args=(*data_kappas_gammas, *relations_coefficients),
        atol=1e-14,
        rtol=1e-10,
        max_num_steps=0,
        t_eval=x_array,
        rk_method=1,
    )
    return I_triplets


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
            atol=1e-14,
            rtol=1e-10,
            max_num_steps=0,
            # first_step=1,
            t_eval=x_array,
            rk_method=1,
        )
    return I_triplets
