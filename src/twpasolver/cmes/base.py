"""Coupled Mode Equations models and solvers optimized for Numba."""

import numba as nb
import numpy as np
from CyRK import nbsolve_ivp

from twpasolver.bonus_types import ComplexArray, FloatArray, nb_complex1d, nb_int2d, nb_float1d
from twpasolver.cmes.solver_config import *


@nb.njit(
    nb_complex1d(
        nb.float64,
        nb_complex1d,
        nb_float1d,
        nb_float1d,
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
    alphas,
    kappas,
    relations_3wm,
    relations_4wm,
    coeffs_3wm,
    coeffs_4wm,
):
    """Return derivatives of Coupled Mode Equations system including arbitrary 3WM and 4WM relations."""
    num_modes = len(currents)
    derivs = np.zeros(num_modes, nb.complex128)
    exp_pos = np.exp(1j * kappas * x)

    alphas_rhs = np.zeros(2 * num_modes, dtype=np.complex128)
    for i in range(num_modes):
        alphas_rhs[i] = currents[i] * exp_pos[i]
        alphas_rhs[i + num_modes] = np.conj(alphas_rhs[i])

    for idx, idx1, idx2 in relations_3wm:
        derivs[idx] += coeffs_3wm[idx] * alphas_rhs[idx1] * alphas_rhs[idx2]
    for idx, idx1, idx2, idx3 in relations_4wm:
        derivs[idx] += (
            coeffs_4wm[idx] * alphas_rhs[idx1] * alphas_rhs[idx2] * alphas_rhs[idx3]
        )
    derivs = (alphas + 1j*kappas) * derivs / exp_pos + alphas * currents
    return derivs


@nb.njit(parallel=True)
def no_reflections_cmes_solve(
    x_array: FloatArray,
    y0_broadcast: ComplexArray,
    gammas_array: ComplexArray,
    relations_3wm: nb_int2d,
    relations_4wm: nb_int2d,
    coeffs_3wm: nb_complex1d,
    coeffs_4wm: nb_complex1d,
) -> ComplexArray:
    """
    Parallel inner loop: solve no_reflections_cmes for every frequency point.

    Separated from the Python wrapper so that nb.prange executes in parallel
    (requires @nb.njit(parallel=True)).

    Args:
        x_array: Output positions, shape (n_positions,).
        y0_broadcast: Initial amplitudes, shape (n_freq, n_modes).
        gammas_array: Propagation constants 1j*kappa - alpha, shape (n_freq, n_modes).
        relations_3wm: 3WM index table, shape (n_3wm, 3).
        relations_4wm: 4WM index table, shape (n_4wm, 4).
        coeffs_3wm: 3WM coefficients, length n_3wm.
        coeffs_4wm: 4WM coefficients, length n_4wm.

    Returns:
        Amplitudes, shape (n_freq, n_modes, n_positions).
    """
    n_freq = y0_broadcast.shape[0]
    n_modes = y0_broadcast.shape[1]
    x_span = (x_array[0], x_array[-1])

    I_tuples = np.empty((n_freq, n_modes, len(x_array)), dtype=nb.complex128)

    for i in nb.prange(n_freq):
        result = nbsolve_ivp(
            no_reflections_cmes,
            x_span,
            y0_broadcast[i],
            args=(
                gammas_array[i].real,
                gammas_array[i].imag,
                relations_3wm,
                relations_4wm,
                coeffs_3wm,
                coeffs_4wm,
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
