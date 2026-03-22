"""Coupled Mode Equations models and solvers optimized for Numba."""

import numba as nb
import numpy as np
from CyRK import nbsolve_ivp

from twpasolver.bonus_types import ComplexArray, FloatArray, nb_complex1d, nb_int2d
from twpasolver.cmes.solver_config import *


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
def basic_3wm_cmes(
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

    abs_y0_sq = abs(y[0]) ** 2
    abs_y1_sq = abs(y[1]) ** 2
    abs_y2_sq = abs(y[2]) ** 2

    pp = abs_y0_sq + 2 * abs_y1_sq + 2 * abs_y2_sq
    ss = 2 * abs_y0_sq + abs_y1_sq + 2 * abs_y2_sq
    ii = 2 * abs_y0_sq + 2 * abs_y1_sq + abs_y2_sq

    derivs = np.empty(3, dtype=np.complex128)
    derivs[0] = a * kp * y[1] * y[2] * np.exp(-1j * dk * t) + b * kp * y[0] * pp
    derivs[1] = a * ks * y[0] * np.conj(y[2]) * np.exp(1j * dk * t) + b * ks * y[1] * ss
    derivs[2] = a * ki * y[0] * np.conj(y[1]) * np.exp(1j * dk * t) + b * ki * y[2] * ii

    return derivs


@nb.njit(parallel=True)
def basic_3wm_cmes_solve(
    k_signal: FloatArray,
    k_idler: FloatArray,
    x_array: FloatArray,
    y0: ComplexArray,
    k_pump: float,
    xi: float,
    epsi: float,
) -> ComplexArray:
    """
    Solve coupled mode equations for multiple frequencies using the standard 3WM model.

    Args:
        k_signal (FloatArray): Array of signal wave numbers.
        k_idler (FloatArray): Array of idler wave numbers.
        x_array (FloatArray): Array of position values for evaluation.
        y0 (ComplexArray): Initial conditions for the coupled mode equations (1D array for single condition or 2D array for multiple).
        k_pump (float): Pump wave number.
        xi (float): Nonlinear coefficient.
        epsi (float): Small perturbation parameter.

    Returns:
        ComplexArray: Solution of the coupled mode equations for the given frequencies.
    """
    len_k = len(k_signal)

    # Handle initial conditions - broadcast if 1D
    if y0.ndim == 1:
        y0_broadcast = np.repeat(y0, len_k).reshape((-1, len_k)).transpose()
    else:
        y0_broadcast = y0

    x_span = (x_array[0], x_array[-1])
    I_tuples = np.empty((len_k, 3, len(x_array)), dtype=np.complex128)

    for i in nb.prange(len_k):
        result = nbsolve_ivp(
            basic_3wm_cmes,
            x_span,
            y0_broadcast[i],
            args=(
                k_pump,
                k_signal[i],
                k_idler[i],
                xi,
                epsi,
            ),
            atol=SOLVER_ATOL,
            rtol=SOLVER_RTOL,
            max_num_steps=SOLVER_MAX_STEPS,
            t_eval=x_array,
            rk_method=SOLVER_RK_METHOD,
            first_step=SOLVER_FIRST_STEP,
        )
        I_tuples[i] = result.y
    return I_tuples


@nb.njit(
    nb_complex1d(
        nb.float64,
        nb_complex1d,
        nb_complex1d,
        nb_int2d,
        nb_int2d,
        nb_complex1d,
        nb_complex1d,
    ),
    cache=True,
)
def no_reflections_cmes(
    x,
    currents,
    gammas,
    relations_3wm,
    relations_4wm,
    coeffs_3wm,
    coeffs_4wm,
):
    """Return derivatives of Coupled Mode Equations system including arbitrary 3WM and 4WM relations."""
    num_modes = len(currents)
    derivs = np.zeros(num_modes, nb.complex128)
    exp_pos = np.exp(gammas * x)

    alphas_rhs = np.zeros(4 * num_modes, dtype=np.complex128)
    for i in range(num_modes):
        alphas_rhs[i] = currents[i] * exp_pos[i]
        alphas_rhs[i + num_modes] = np.conj(alphas_rhs[i])

    for idx, idx1, idx2 in relations_3wm:
        derivs[idx] += coeffs_3wm[idx] * alphas_rhs[idx1] * alphas_rhs[idx2]
    for idx, idx1, idx2, idx3 in relations_4wm:
        derivs[idx] += (
            coeffs_4wm[idx] * alphas_rhs[idx1] * alphas_rhs[idx2] * alphas_rhs[idx3]
        )
    derivs[:num_modes] = gammas * derivs[:num_modes] / exp_pos
    return derivs


def cme_general_solve_freq_array(
    x_array: FloatArray,
    y0_array: ComplexArray,
    data_kappas_gammas_array,
    relations_coefficients,
    thin: int = 200,
    reflections: bool = True,
    with_loss: bool = False,
) -> ComplexArray:
    """
    Solve coupled mode equations for multiple frequency points using numba prange.

    Args:
        x_array (FloatArray): Array of position values for evaluation.
        y0_array (ComplexArray): Initial currents for each mode and frequency (2D array: [n_freq, n_modes] or 1D array for single condition).
        data_kappas_gammas_array: List of parameter arrays for each frequency point.
        relations_coefficients: tables with indexes representing mixing relations between modes and relative coefficients.
        thin (int): Thinning factor for plotting.

    Returns:
        ComplexArray: Solution of the coupled mode equations for all frequencies (3D array: [n_freq, n_modes, n_positions]).
    """
    n_freq = len(data_kappas_gammas_array)
    n_modes = len(data_kappas_gammas_array[0][0])  # kappas is always the first element

    # Convert relations_coefficients to numpy arrays with proper dtypes
    relations_3wm, relations_4wm, coeffs_3wm, coeffs_4wm = relations_coefficients
    relations_3wm = np.array(relations_3wm, dtype=np.int64)
    relations_4wm = np.array(relations_4wm, dtype=np.int64)
    coeffs_3wm = np.array(coeffs_3wm, dtype=np.complex128)
    coeffs_4wm = np.array(coeffs_4wm, dtype=np.complex128)

    # Ideal model
    args_array = []
    for i in range(n_freq):
        kappas = np.array(data_kappas_gammas_array[i][0], dtype=np.float64)
        args_array.append(
            (kappas, relations_3wm, relations_4wm, coeffs_3wm, coeffs_4wm)
        )
    n_freq = len(args_array)
    # Handle initial conditions - broadcast if 1D
    if y0_array.ndim == 1:
        y0_broadcast = np.repeat(y0_array, n_freq).reshape((-1, n_freq)).transpose()
    else:
        y0_broadcast = y0_array
    I_tuples = np.empty((n_freq, n_modes, len(x_array)), dtype=np.complex128)
    # Use parallel loop for frequency iteration
    for i in nb.prange(n_freq):
        x_span = (x_array[0], x_array[-1])

        result = nbsolve_ivp(
            no_reflections_cmes,
            x_span,
            y0_broadcast[i],
            args=args_array[i],
            atol=SOLVER_ATOL,
            rtol=SOLVER_RTOL,
            max_num_steps=SOLVER_MAX_STEPS,
            t_eval=x_array,
            rk_method=SOLVER_RK_METHOD,
            first_step=SOLVER_FIRST_STEP,
        )
        I_tuples[i] = result.y
    return I_tuples
