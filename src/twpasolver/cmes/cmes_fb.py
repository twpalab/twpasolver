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
from twpasolver.cmes.solver_config import *


@nb.njit
def _filter_wm_terms(kappas, relations_3wm, relations_4wm, relative_deltak_cutoff):
    num_modes = len(kappas)
    len_3wm = len(relations_3wm)
    len_4wm = len(relations_4wm)

    # Pre-allocate with exact maximum size
    valid_3wm = np.empty((len_3wm * 4, 3), dtype=relations_3wm.dtype)
    valid_4wm = np.empty((len_4wm * 8, 4), dtype=relations_4wm.dtype)
    num_valid_3wm = 0
    num_valid_4wm = 0

    # Pre-compute kappa sign arrays for all combinations
    signs_3wm = np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]], dtype=np.int8)
    signs_4wm = np.array(
        [
            [1, 1, 1],
            [1, 1, -1],
            [1, -1, 1],
            [1, -1, -1],
            [-1, 1, 1],
            [-1, 1, -1],
            [-1, -1, 1],
            [-1, -1, -1],
        ],
        dtype=np.int8,
    )

    # Process 3WM terms
    for k3wm in range(len_3wm):
        rel_3wm = relations_3wm[k3wm].copy()
        base_kappa = kappas[rel_3wm[0]]

        # Pre-extract mode indices and compute base kappas
        mode1 = rel_3wm[1]
        mode2 = rel_3wm[2]

        # Determine if modes are conjugated (>= num_modes)
        is_conj1 = mode1 >= num_modes
        is_conj2 = mode2 >= num_modes

        # Get actual kappa indices
        k1 = mode1 - num_modes if is_conj1 else mode1
        k2 = mode2 - num_modes if is_conj2 else mode2

        kappa1 = kappas[k1]
        kappa2 = kappas[k2]

        for sign_idx in range(4):
            s1, s2 = signs_3wm[sign_idx]

            # Compute delta kappa more efficiently
            dkappa = -base_kappa
            dkappa += s1 * kappa1 * (-1 if is_conj1 else 1)
            dkappa += s2 * kappa2 * (-1 if is_conj2 else 1)

            if np.abs(dkappa / base_kappa) < relative_deltak_cutoff:
                # Create modified relation
                valid_3wm[num_valid_3wm, 0] = rel_3wm[0]
                valid_3wm[num_valid_3wm, 1] = rel_3wm[1] + (
                    2 * num_modes if s1 == -1 else 0
                )
                valid_3wm[num_valid_3wm, 2] = rel_3wm[2] + (
                    2 * num_modes if s2 == -1 else 0
                )
                num_valid_3wm += 1

    # Process 4WM terms
    for k4wm in range(len_4wm):
        rel_4wm = relations_4wm[k4wm].copy()
        base_kappa = kappas[rel_4wm[0]]

        # Pre-extract mode indices
        mode1 = rel_4wm[1]
        mode2 = rel_4wm[2]
        mode3 = rel_4wm[3]

        # Determine conjugation
        is_conj1 = mode1 >= num_modes
        is_conj2 = mode2 >= num_modes
        is_conj3 = mode3 >= num_modes

        # Get actual kappa indices
        k1 = mode1 - num_modes if is_conj1 else mode1
        k2 = mode2 - num_modes if is_conj2 else mode2
        k3 = mode3 - num_modes if is_conj3 else mode3

        kappa1 = kappas[k1]
        kappa2 = kappas[k2]
        kappa3 = kappas[k3]

        for sign_idx in range(8):
            s1, s2, s3 = signs_4wm[sign_idx]

            # Compute delta kappa
            dkappa = -base_kappa
            dkappa += s1 * kappa1 * (-1 if is_conj1 else 1)
            dkappa += s2 * kappa2 * (-1 if is_conj2 else 1)
            dkappa += s3 * kappa3 * (-1 if is_conj3 else 1)

            if np.abs(dkappa / base_kappa) < relative_deltak_cutoff:
                # Create modified relation
                valid_4wm[num_valid_4wm, 0] = rel_4wm[0]
                valid_4wm[num_valid_4wm, 1] = rel_4wm[1] + (
                    2 * num_modes if s1 == -1 else 0
                )
                valid_4wm[num_valid_4wm, 2] = rel_4wm[2] + (
                    2 * num_modes if s2 == -1 else 0
                )
                valid_4wm[num_valid_4wm, 3] = rel_4wm[3] + (
                    2 * num_modes if s3 == -1 else 0
                )
                num_valid_4wm += 1

    return valid_3wm[:num_valid_3wm], valid_4wm[:num_valid_4wm]


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
    # return derivs
    derivs[:num_modes] = gammas * derivs[:num_modes] / exp_pos
    derivs[num_modes : 2 * num_modes] = derivs[:num_modes]
    derivs[2 * num_modes :] = currents
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
def general_cme_with_backward_cutoff(
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

    alphas_rhs = np.empty(4 * num_modes, dtype=np.complex128)
    for i in range(num_modes):
        alphas_rhs[i] = currents[i] * exp_pos[i]
        alphas_rhs[i + num_modes] = np.conj(alphas_rhs[i])
        alphas_rhs[i + 2 * num_modes] = (
            np.interp(x, x_evals, currents_evals[i].real)
            + 1j * np.interp(x, x_evals, currents_evals[i].imag)
        ) * exp_neg[i]
        alphas_rhs[i + 3 * num_modes] = np.conj(alphas_rhs[i + 2 * num_modes])

    for idx, idx1, idx2 in relations_3wm:
        derivs[idx] += coeffs_3wm[idx] * alphas_rhs[idx1] * alphas_rhs[idx2]
    for idx, idx1, idx2, idx3 in relations_4wm:
        derivs[idx] += (
            coeffs_4wm[idx] * alphas_rhs[idx1] * alphas_rhs[idx2] * alphas_rhs[idx3]
        )
    # return derivs
    for i in range(num_modes):
        di = gammas[i] * derivs[i]
        derivs[i] = di / exp_pos[i]
        derivs[num_modes + i] = derivs[i]
        derivs[2 * num_modes + i] = currents[i]
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
        derivs[i] = di / exp_pos[i]  # / gammas[i]
        derivs[num_modes + i] = derivs[i]  # * exp_pos[i]
        derivs[2 * num_modes + i] = currents[i]  # * exp_pos[i]
    return derivs


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
def cme_general_solve_freq_array_fb_cutoff(
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
    cutoff: float = 0.1,
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
    Z0 = 50
    Zb = Z0 * (1 + reflections_array) / (1 - reflections_array)
    for i in nb.prange(n_freq):
        A_bwd = np.empty(
            (n_modes, len(x_array)), dtype=np.complex128
        )  # Required for proper compilation
        A_fwd = np.empty(
            (n_modes, len(x_array)), dtype=np.complex128
        )  # Required for proper compilation
        I_init_fwd = y0_array_fwd * transmission_in[i]
        relations_3wm_optimized, relations_4wm_optimized = _filter_wm_terms(
            gammas_array[i].imag, relations_3wm, relations_4wm, cutoff
        )
        for k in range(n_passes):
            if k == 0:
                result = nbsolve_ivp(
                    general_cme_first_round,
                    x_span,
                    I_init_fwd,
                    args=(
                        gammas_array[i],
                        relations_3wm_optimized,
                        relations_4wm_optimized,
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
                A_fwd_end = A_fwd[:n_modes, -1] * np.exp(gammas_array[i] * x_end)
                I_init_bwd = (
                    y0_array_bwd * transmission_in[i] + reflections_array[i] * A_fwd_end
                )
                result = nbsolve_ivp(
                    general_cme_with_backward_cutoff,
                    x_span,
                    I_init_bwd,
                    args=(
                        gammas_array[i],
                        relations_3wm_optimized,
                        relations_4wm_optimized,
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
                    general_cme_with_backward_cutoff,
                    x_span,
                    I_init_fwd,
                    args=(
                        gammas_array[i],
                        relations_3wm_optimized,
                        relations_4wm_optimized,
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
                A_fwd_end = A_fwd[:n_modes, -1] * np.exp(gammas_array[i] * x_end)
                I_init_bwd = (
                    y0_array_bwd * transmission_in[i] + reflection_coeff * A_fwd_end
                )
                result = nbsolve_ivp(
                    general_cme_with_backward_cutoff,
                    x_span,
                    I_init_bwd,
                    args=(
                        gammas_array[i],
                        relations_3wm_optimized,
                        relations_4wm_optimized,
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
            A_bwd_end = A_bwd[:n_modes, -1] * np.exp(gammas_array[i] * x_end)
            I_init_fwd = (
                y0_array_fwd * transmission_in[i] + reflection_coeff * A_bwd_end
            )
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
    cutoff: float = 0.1,
    use_cutoff: bool = True,
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
    if use_cutoff:
        return cme_general_solve_freq_array_fb_cutoff(
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
            cutoff,
        )
    else:
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
