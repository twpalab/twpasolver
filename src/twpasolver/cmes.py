"""Coupled Mode Equations models and solvers optimized for Numba."""

import numba as nb
import numpy as np
from CyRK import nbsolve_ivp

from twpasolver.bonus_types import (
    ComplexArray,
    FloatArray,
    nb_complex1d,
    nb_complex2d,
    nb_float1d,
    nb_int2d,
)

# Global solver configuration
SOLVER_ATOL = 1e-14
SOLVER_RTOL = 1e-10
SOLVER_MAX_STEPS = 0
SOLVER_RK_METHOD = 1
SOLVER_FIRST_STEP = 10


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


@nb.njit(
    nb_complex1d(
        nb.float64,
        nb_complex1d,
        nb_float1d,  # kappas as float64
        nb_int2d,
        nb_int2d,
        nb_complex1d,  # coeffs_3wm should be complex
        nb_complex1d,  # coeffs_4wm should be complex
    ),
    cache=True,
)
def general_cme_ideal(
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
        nb_complex1d,
        nb_int2d,
        nb_int2d,
        nb_complex1d,
        nb_complex1d,
    ),
    cache=True,
)
def general_cme_first_round(
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
    derivs = np.zeros(num_modes * 3, nb.complex128)
    exp_pos = np.exp(gammas * x)

    alphas_rhs = np.empty(2 * num_modes, dtype=np.complex128)
    for i in range(num_modes):
        alphas_rhs[i] = currents[i] * exp_pos[i]
        alphas_rhs[i + num_modes] = np.conj(alphas_rhs[i])

    for idx, idx1, idx2 in relations_3wm:
        derivs[idx] += coeffs_3wm[idx] * alphas_rhs[idx1] * alphas_rhs[idx2]
    for idx, idx1, idx2, idx3 in relations_4wm:
        derivs[idx] += (
            coeffs_4wm[idx] * alphas_rhs[idx1] * alphas_rhs[idx2] * alphas_rhs[idx3]
        )
    # return derivs
    derivs[:num_modes] = gammas * derivs[:num_modes] / exp_pos
    derivs[num_modes : 2 * num_modes] = derivs[:num_modes]
    derivs[2 * num_modes :] = currents  # * exp_pos
    return derivs


@nb.njit(
    nb_complex1d(
        nb.float64,
        nb_complex1d,
        nb_complex1d,
        nb_int2d,
        nb_int2d,
        nb_complex1d,
        nb_complex1d,
        nb_float1d,
        nb_complex2d,
        nb_complex2d,
        nb.float64,
    ),
    cache=True,
)
def general_cme_with_backward(
    x,
    currents,
    gammas,
    relations_3wm,
    relations_4wm,
    coeffs_3wm,
    coeffs_4wm,
    x_evals,
    dcurrents_evals,
    currents_evals,
    end,
):
    """Return derivatives of Coupled Mode Equations system including arbitrary 3WM and 4WM relations."""
    num_modes = len(currents)
    derivs = np.zeros(3 * num_modes, nb.complex128)
    exp_pos = np.exp(gammas * x)
    exp_neg = np.exp(gammas * (end - x))

    alphas_rhs = np.empty(2 * num_modes, dtype=np.complex128)
    for i in range(num_modes):
        alphas_rhs[i] = (
            currents[i] * exp_pos[i]
            + (
                np.interp(x, x_evals, currents_evals[i].real)
                + 1j * np.interp(x, x_evals, currents_evals[i].imag)
            )
            * exp_neg[i]
        )
        alphas_rhs[i + num_modes] = np.conj(alphas_rhs[i])

    for idx, idx1, idx2 in relations_3wm:
        derivs[idx] += coeffs_3wm[idx] * alphas_rhs[idx1] * alphas_rhs[idx2]
    for idx, idx1, idx2, idx3 in relations_4wm:
        derivs[idx] += (
            coeffs_4wm[idx] * alphas_rhs[idx1] * alphas_rhs[idx2] * alphas_rhs[idx3]
        )
    # return derivs
    for i in range(num_modes):
        dc_interp = np.interp(x, x_evals, dcurrents_evals[i].real) + 1j * np.interp(
            x, x_evals, dcurrents_evals[i].imag
        )
        di = gammas[i] * derivs[i] - dc_interp * exp_neg[i]
        derivs[i] = di /exp_pos[i]  # / gammas[i]
        derivs[num_modes + i] = derivs[i]  # * exp_pos[i]
        derivs[2 * num_modes + i] = currents[i]  # * exp_pos[i]
    return derivs


@nb.njit(
    nb_complex1d(
        nb.float64,
        nb_complex1d,
        nb_float1d,
        nb_float1d,  # alphas as float64
        nb_int2d,
        nb_int2d,
        nb_complex1d,  # coeffs_3wm should be complex
        nb_complex1d,  # coeffs_4wm should be complex
    ),
    cache=True,
)
def general_cme_loss_only(
    x,
    currents,
    kappas,
    alphas,
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
    return (1j * kappas - alphas) * derivs /exp_pos - alphas * currents


@nb.njit(
    nb_complex1d(
        nb.float64,
        nb_complex1d,
        nb_float1d,  # kappas as float64
        nb_float1d,  # alphas as float64
        nb_complex1d,
        nb_complex1d,
        nb_int2d,
        nb_int2d,
        nb_complex1d,  # coeffs_3wm should be complex
        nb_complex1d,  # coeffs_4wm should be complex
    ),
    cache=True,
)
def general_cme(
    x,
    currents,
    kappas,
    alphas,
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
    common_term_neg = ts_reflection_neg * np.exp(-1j * kappas * x)
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

    return ((1j * kappas) * derivs) / (common_term_pos - common_term_neg)


@nb.njit
def _solve_single_frequency_general(
    x_array: FloatArray,
    y0: ComplexArray,
    cme_model,  # Function reference
    args: tuple,  # Arguments for the CME model
) -> ComplexArray:
    """Solve CME for a single frequency point using nbsolve_ivp."""
    x_span = (x_array[0], x_array[-1])

    result = nbsolve_ivp(
        cme_model,
        x_span,
        y0,
        args=args,
        atol=SOLVER_ATOL,
        rtol=SOLVER_RTOL,
        max_num_steps=SOLVER_MAX_STEPS,
        t_eval=x_array,
        rk_method=SOLVER_RK_METHOD,
        first_step=SOLVER_FIRST_STEP,
    )
    return result.y


@nb.njit(parallel=True)
def _solve_multiple_frequencies_general(
    x_array: FloatArray,
    y0_array: ComplexArray,
    cme_model,  # Function reference
    args_array,  # Array of argument tuples for each frequency
    n_modes: int,
) -> ComplexArray:
    """Solve CMEs for multiple frequency points using numba prange."""
    n_freq = len(args_array)

    # Handle initial conditions - broadcast if 1D
    if y0_array.ndim == 1:
        y0_broadcast = np.repeat(y0_array, n_freq).reshape((-1, n_freq)).transpose()
    else:
        y0_broadcast = y0_array

    I_triplets = np.empty((n_freq, n_modes, len(x_array)), dtype=np.complex128)

    # Use parallel loop for frequency iteration
    for i in nb.prange(n_freq):
        result = _solve_single_frequency_general(
            x_array, y0_broadcast[i], cme_model, args_array[i]
        )
        I_triplets[i] = result

    return I_triplets


@nb.njit(parallel=True)
def cme_general_solve_freq_array_fb(
    x_array: FloatArray,
    y0_array_fwd: ComplexArray,
    y0_array_bwd: ComplexArray,
    gammas_array: ComplexArray,
    reflections_array: ComplexArray,
    relations_3wm,
    relations_4wm,
    coeffs_3wm,
    coeffs_4wm,
    thin: int = 200,
    n_passes: int = 3,
) -> ComplexArray:
    """Solve full general CMEs (with reflections) for multiple frequency points using numba prange."""
    n_freq = gammas_array.shape[0]
    n_modes = gammas_array.shape[1]
    transmission_in = 1 - reflections_array

    x_span = (x_array[0], x_array[-1])
    x_end = x_array[-1]
    # Modified to store results for all passes
    I_tuples = np.empty(
        (2 * n_freq * n_passes, n_modes, len(x_array)), dtype=np.complex128
    )
    print(relations_3wm, relations_4wm)
    for i in nb.prange(n_freq):
        A_bwd = np.empty(
            (n_modes, len(x_array)), dtype=np.complex128
        )  # Required for proper compilation
        A_fwd = np.empty(
            (n_modes, len(x_array)), dtype=np.complex128
        )  # Required for proper compilation
        I_init_fwd = y0_array_fwd * transmission_in[i]

        for k in range(n_passes):
            if k == 0:
                result = nbsolve_ivp(
                    general_cme_first_round,
                    x_span,
                    I_init_fwd,
                    args=(
                        gammas_array[i],
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
                    capture_extra=True,
                )
                A_fwd = result.y

                I_init_bwd = y0_array_bwd * transmission_in[i] + reflections_array[
                    i
                ] * A_fwd[:n_modes, -1] * np.exp(gammas_array[i] * x_end)
                result = nbsolve_ivp(
                    general_cme_with_backward,
                    x_span,
                    I_init_bwd,
                    args=(
                        gammas_array[i],
                        relations_3wm,
                        relations_4wm,
                        coeffs_3wm,
                        coeffs_4wm,
                        x_array,
                        A_fwd[n_modes : 2 * n_modes, ::-1],
                        A_fwd[:n_modes, ::-1],
                        x_end,
                    ),
                    atol=SOLVER_ATOL,
                    rtol=SOLVER_RTOL,
                    max_num_steps=SOLVER_MAX_STEPS,
                    t_eval=x_array,
                    rk_method=SOLVER_RK_METHOD,
                    first_step=SOLVER_FIRST_STEP,
                    capture_extra=True,
                )
                A_bwd = result.y

            else:
                result = nbsolve_ivp(
                    general_cme_with_backward,
                    x_span,
                    I_init_fwd,
                    args=(
                        gammas_array[i],
                        relations_3wm,
                        relations_4wm,
                        coeffs_3wm,
                        coeffs_4wm,
                        x_array,
                        A_bwd[n_modes : 2 * n_modes, ::-1],
                        A_bwd[:n_modes, ::-1],
                        x_end,
                    ),
                    atol=SOLVER_ATOL,
                    rtol=SOLVER_RTOL,
                    max_num_steps=SOLVER_MAX_STEPS,
                    t_eval=x_array,
                    rk_method=SOLVER_RK_METHOD,
                    first_step=SOLVER_FIRST_STEP,
                    capture_extra=True,
                )
                A_fwd = result.y

                # Store forward pass results for this pass

                I_init_bwd = y0_array_bwd * transmission_in[i] + reflections_array[
                    i
                ] * A_fwd[:n_modes, -1] * np.exp(gammas_array[i] * x_end)
                result = nbsolve_ivp(
                    general_cme_with_backward,
                    x_span,
                    I_init_bwd,
                    args=(
                        gammas_array[i],
                        relations_3wm,
                        relations_4wm,
                        coeffs_3wm,
                        coeffs_4wm,
                        x_array,
                        A_fwd[n_modes : 2 * n_modes, ::-1],
                        A_fwd[:n_modes, ::-1],
                        x_end,
                    ),
                    atol=SOLVER_ATOL,
                    rtol=SOLVER_RTOL,
                    max_num_steps=SOLVER_MAX_STEPS,
                    t_eval=x_array,
                    rk_method=SOLVER_RK_METHOD,
                    first_step=SOLVER_FIRST_STEP,
                    capture_extra=True,
                )
                A_bwd = result.y

            # Store backward pass results for this pass

            I_init_fwd = y0_array_fwd * transmission_in[i] + reflections_array[
                i
            ] * A_bwd[:n_modes, -1] * np.exp(gammas_array[i] * x_end)
            I_tuples[i + k * n_freq] = A_fwd[2 * n_modes :, :]
            I_tuples[i + k * n_freq + n_freq * n_passes] = A_bwd[2 * n_modes :, :]

    return I_tuples 


@nb.njit(parallel=True)
def cme_general_solve_freq_array_fb_old(
    x_array: FloatArray,
    y0_array_fwd: ComplexArray,
    y0_array_bwd: ComplexArray,
    gammas_array: ComplexArray,
    reflections_array: ComplexArray,
    relations_3wm,
    relations_4wm,
    coeffs_3wm,
    coeffs_4wm,
    thin: int = 200,
    n_passes: int = 3,
) -> ComplexArray:
    """Solve full general CMEs (with reflections) for multiple frequency points using numba prange."""
    n_freq = gammas_array.shape[0]
    n_modes = gammas_array.shape[1]
    transmission_in = 1 - reflections_array

    x_span = (x_array[0], x_array[-1])
    x_end = x_array[-1]
    # Modified to store results for all passes
    I_tuples = np.empty(
        (2 * n_freq * n_passes, n_modes, len(x_array)), dtype=np.complex128
    )

    for i in nb.prange(n_freq):
        A_bwd = np.empty(
            (n_modes, len(x_array)), dtype=np.complex128
        )  # Required for proper compilation
        A_fwd = np.empty(
            (n_modes, len(x_array)), dtype=np.complex128
        )  # Required for proper compilation
        I_init_fwd = y0_array_fwd * transmission_in[i]

        for k in range(n_passes):
            if k == 0:
                result = nbsolve_ivp(
                    general_cme_first_round,
                    x_span,
                    I_init_fwd,
                    args=(
                        gammas_array[i],
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
                    capture_extra=True,
                )
                A_fwd = result.y

                I_init_bwd = y0_array_bwd * transmission_in[i] + reflections_array[
                    i
                ] * A_fwd[:n_modes, -1] * np.exp(gammas_array[i] * x_end)
                result = nbsolve_ivp(
                    general_cme_first_round,
                    x_span,
                    I_init_bwd,
                    args=(
                        gammas_array[i],
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
                    capture_extra=True,
                )
                A_bwd = result.y

            else:
                result = nbsolve_ivp(
                    general_cme_with_backward,
                    x_span,
                    I_init_fwd,
                    args=(
                        gammas_array[i],
                        relations_3wm,
                        relations_4wm,
                        coeffs_3wm,
                        coeffs_4wm,
                        x_array,
                        A_bwd[n_modes : 2 * n_modes, ::-1],
                        A_bwd[2 * n_modes :, ::-1],
                    ),
                    atol=SOLVER_ATOL,
                    rtol=SOLVER_RTOL,
                    max_num_steps=SOLVER_MAX_STEPS,
                    t_eval=x_array,
                    rk_method=SOLVER_RK_METHOD,
                    first_step=SOLVER_FIRST_STEP,
                    capture_extra=True,
                )
                A_fwd = result.y

                # Store forward pass results for this pass

                I_init_bwd = y0_array_bwd * transmission_in[i] + reflections_array[
                    i
                ] * A_fwd[:n_modes, -1] * np.exp(gammas_array[i] * x_end)
                result = nbsolve_ivp(
                    general_cme_with_backward,
                    x_span,
                    I_init_bwd,
                    args=(
                        gammas_array[i],
                        relations_3wm,
                        relations_4wm,
                        coeffs_3wm,
                        coeffs_4wm,
                        x_array,
                        A_fwd[n_modes : 2 * n_modes, ::-1],
                        A_fwd[2 * n_modes :, ::-1],
                    ),
                    atol=SOLVER_ATOL,
                    rtol=SOLVER_RTOL,
                    max_num_steps=SOLVER_MAX_STEPS,
                    t_eval=x_array,
                    rk_method=SOLVER_RK_METHOD,
                    first_step=SOLVER_FIRST_STEP,
                    capture_extra=True,
                )
                A_bwd = result.y

            # Store backward pass results for this pass

            I_init_fwd = y0_array_fwd * transmission_in[i] + reflections_array[
                i
            ] * A_bwd[:n_modes, -1] * np.exp(gammas_array[i] * x_end)
            I_tuples[i + k * n_freq] = A_fwd[2 * n_modes :, :]
            I_tuples[i + k * n_freq + n_freq * n_passes] = A_bwd[2 * n_modes :, :]

    return I_tuples


def cme_solve_forward_backward(
    x_array: FloatArray,
    y0_array_fwd: ComplexArray,
    y0_array_bwd: ComplexArray,
    data_kappas_alpha_reflections_array,
    relations_coefficients,
    thin: int = 200,
    passes: int = 3,
) -> ComplexArray:
    """Get derivatives of CMEs with pre-computed current amplitudes in the opposite direction."""
    n_freq = len(data_kappas_alpha_reflections_array)
    n_modes = len(data_kappas_alpha_reflections_array[0][0])

    # Convert relations_coefficients to numpy arrays with proper dtypes
    relations_3wm, relations_4wm, coeffs_3wm, coeffs_4wm = relations_coefficients
    relations_3wm = np.array(relations_3wm, dtype=np.int64)
    relations_4wm = np.array(relations_4wm, dtype=np.int64)
    coeffs_3wm = np.array(coeffs_3wm, dtype=np.complex128)  # Ensure complex type
    coeffs_4wm = np.array(coeffs_4wm, dtype=np.complex128)  # Ensure complex type

    gammas_array = np.empty((n_freq, n_modes), dtype=np.complex128)
    reflections_array = np.empty((n_freq, n_modes), dtype=np.complex128)
    for i in range(n_freq):
        gammas_array[i] = np.array(
            1j * data_kappas_alpha_reflections_array[i][0]
            - data_kappas_alpha_reflections_array[i][1],
            dtype=np.complex128,
        )
        reflections_array[i] = np.array(
            data_kappas_alpha_reflections_array[i][2], dtype=np.complex128
        )

    return cme_general_solve_freq_array_fb(
        x_array,
        y0_array_fwd,
        y0_array_bwd,
        gammas_array,
        reflections_array,
        relations_3wm,
        relations_4wm,
        coeffs_3wm,
        coeffs_4wm,
        thin,
        passes,
    )


def cme_general_solve_freq_array(
    x_array: FloatArray,
    y0_array: ComplexArray,
    data_kappas_gammas_array,  # Can't use specific type annotation due to heterogeneous data
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
        reflections (bool): Whether to include reflections.
        with_loss (bool): Whether to use the loss-only model.

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

    # # Route to appropriate solver based on model complexity
    # if with_loss and not reflections:
    #     # Loss-only model
    #     args_array = []
    #     for i in range(n_freq):
    #         kappas = np.array(data_kappas_gammas_array[i][0], dtype=np.float64)
    #         alphas = np.array(data_kappas_gammas_array[i][1], dtype=np.float64)
    #         args_array.append(
    #             (kappas, alphas, relations_3wm, relations_4wm, coeffs_3wm, coeffs_4wm)
    #         )

    #     return _solve_multiple_frequencies_general(
    #         x_array, y0_array, general_cme_loss_only, args_array, n_modes
    #     )

    # elif reflections and not with_loss:
    # Full model with reflections
    args_array = []
    for i in range(n_freq):
        kappas = np.array(data_kappas_gammas_array[i][0], dtype=np.float64)
        alphas = np.array(data_kappas_gammas_array[i][1], dtype=np.float64)
        ts_reflection = np.array(data_kappas_gammas_array[i][2], dtype=np.complex128)
        ts_reflection_neg = np.array(
            data_kappas_gammas_array[i][3], dtype=np.complex128
        )
        args_array.append(
            (
                kappas,
                alphas,
                ts_reflection,
                ts_reflection_neg,
                relations_3wm,
                relations_4wm,
                coeffs_3wm,
                coeffs_4wm,
            )
        )
    return _solve_multiple_frequencies_general(
        x_array, y0_array, general_cme, args_array, n_modes
    )

    # else:
    #     # Ideal model
    #     args_array = []
    #     for i in range(n_freq):
    #         kappas = np.array(data_kappas_gammas_array[i][0], dtype=np.float64)
    #         args_array.append(
    #             (kappas, relations_3wm, relations_4wm, coeffs_3wm, coeffs_4wm)
    #         )

    #     return _solve_multiple_frequencies_general(
    #         x_array, y0_array, general_cme_ideal, args_array, n_modes
    #     )


@nb.njit(parallel=True)
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
    I_triplets = np.empty((len_k, 3, len(x_array)), dtype=np.complex128)

    for i in nb.prange(len_k):
        result = nbsolve_ivp(
            CMEode_complete,
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
        I_triplets[i] = result.y
    return I_triplets


def configure_solver(
    atol=None, rtol=None, max_steps=None, rk_method=None, first_step=None
):
    """
    Configure global solver parameters.

    Args:
        atol (float, optional): Absolute tolerance for the solver
        rtol (float, optional): Relative tolerance for the solver
        max_steps (int, optional): Maximum number of steps (0 = unlimited)
        rk_method (int, optional): Runge-Kutta method (1=RK45, 2=DOP853, etc.)
        first_step (float, optional): Initial step size
    """
    global SOLVER_ATOL, SOLVER_RTOL, SOLVER_MAX_STEPS, SOLVER_RK_METHOD, SOLVER_FIRST_STEP

    if atol is not None:
        SOLVER_ATOL = atol
    if rtol is not None:
        SOLVER_RTOL = rtol
    if max_steps is not None:
        SOLVER_MAX_STEPS = max_steps
    if rk_method is not None:
        SOLVER_RK_METHOD = rk_method
    if first_step is not None:
        SOLVER_FIRST_STEP = first_step


def get_solver_config():
    """
    Get current solver configuration.

    Returns:
        dict: Dictionary with current solver parameters
    """
    return {
        "atol": SOLVER_ATOL,
        "rtol": SOLVER_RTOL,
        "max_steps": SOLVER_MAX_STEPS,
        "rk_method": SOLVER_RK_METHOD,
        "first_step": SOLVER_FIRST_STEP,
    }
