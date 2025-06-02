"""Coupled Mode Equations models and solvers optimized for Numba."""

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
        nb_float1d,  # kappas as float64
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
    return (1j * kappas - alphas) * derivs * np.conj(exp_pos) - alphas * currents


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

    return ((1j * kappas - alphas) * derivs - alphas * alphas_rhs[:num_modes]) / (
        common_term_pos - common_term_neg
    )


@nb.njit(parallel=True)
def cme_general_solve_freq_array_ideal(
    x_array: FloatArray,
    y0_array: ComplexArray,
    kappas_array: FloatArray,  # 2D array: [n_freq, n_modes] - float64
    relations_3wm,
    relations_4wm,
    coeffs_3wm,
    coeffs_4wm,
    thin: int = 200,
) -> ComplexArray:
    """Solve ideal general CMEs for multiple frequency points using numba prange."""
    cme_model = general_cme_ideal
    n_freq = kappas_array.shape[0]
    n_modes = kappas_array.shape[1]

    # Handle initial conditions - broadcast if 1D
    if y0_array.ndim == 1:
        y0_broadcast = np.repeat(y0_array, n_freq).reshape((-1, n_freq)).transpose()
    else:
        y0_broadcast = y0_array

    x_span = (x_array[0], x_array[-1])
    I_triplets = np.empty((n_freq, n_modes, len(x_array)), dtype=np.complex128)

    # Use parallel loop for frequency iteration
    for i in nb.prange(n_freq):
        _, I_triplets[i], _, _ = nbrk_ode(
            cme_model,
            x_span,
            y0_broadcast[i],
            args=(
                kappas_array[i],
                relations_3wm,
                relations_4wm,
                coeffs_3wm,
                coeffs_4wm,
            ),
            atol=1e-14,
            rtol=1e-10,
            max_num_steps=0,
            t_eval=x_array,
            rk_method=1,
            first_step=1,
        )
    return I_triplets


@nb.njit(parallel=True)
def cme_general_solve_freq_array_loss(
    x_array: FloatArray,
    y0_array: ComplexArray,
    kappas_array: FloatArray,  # 2D array: [n_freq, n_modes] - float64
    alphas_array: FloatArray,  # 2D array: [n_freq, n_modes] - float64
    relations_3wm,
    relations_4wm,
    coeffs_3wm,
    coeffs_4wm,
    thin: int = 200,
) -> ComplexArray:
    """Solve loss-only general CMEs for multiple frequency points using numba prange."""
    cme_model = general_cme_loss_only
    n_freq = kappas_array.shape[0]
    n_modes = kappas_array.shape[1]

    # Handle initial conditions - broadcast if 1D
    if y0_array.ndim == 1:
        y0_broadcast = np.repeat(y0_array, n_freq).reshape((-1, n_freq)).transpose()
    else:
        y0_broadcast = y0_array

    x_span = (x_array[0], x_array[-1])
    I_triplets = np.empty((n_freq, n_modes, len(x_array)), dtype=np.complex128)

    # Use parallel loop for frequency iteration
    for i in nb.prange(n_freq):
        _, I_triplets[i], _, _ = nbrk_ode(
            cme_model,
            x_span,
            y0_broadcast[i],
            args=(
                kappas_array[i],
                alphas_array[i],
                relations_3wm,
                relations_4wm,
                coeffs_3wm,
                coeffs_4wm,
            ),
            atol=1e-14,
            rtol=1e-10,
            max_num_steps=0,
            t_eval=x_array,
            rk_method=1,
            first_step=1,
        )
    return I_triplets


@nb.njit(parallel=True)
def cme_general_solve_freq_array_full(
    x_array: FloatArray,
    y0_array: ComplexArray,
    kappas_array: FloatArray,  # 2D array: [n_freq, n_modes] - float64
    alphas_array: FloatArray,  # 2D array: [n_freq, n_modes] - float64
    ts_reflection_array: ComplexArray,  # 2D array: [n_freq, n_modes]
    ts_reflection_neg_array: ComplexArray,  # 2D array: [n_freq, n_modes]
    relations_3wm,
    relations_4wm,
    coeffs_3wm,
    coeffs_4wm,
    thin: int = 200,
) -> ComplexArray:
    """Solve full general CMEs (with reflections) for multiple frequency points using numba prange."""
    cme_model = general_cme
    n_freq = kappas_array.shape[0]
    n_modes = kappas_array.shape[1]

    # Handle initial conditions - broadcast if 1D
    if y0_array.ndim == 1:
        y0_broadcast = np.repeat(y0_array, n_freq).reshape((-1, n_freq)).transpose()
    else:
        y0_broadcast = y0_array

    x_span = (x_array[0], x_array[-1])
    I_triplets = np.empty((n_freq, n_modes, len(x_array)), dtype=np.complex128)

    # Use parallel loop for frequency iteration
    for i in nb.prange(n_freq):
        _, I_triplets[i], _, _ = nbrk_ode(
            cme_model,
            x_span,
            y0_broadcast[i],
            args=(
                kappas_array[i],
                alphas_array[i],
                ts_reflection_array[i],
                ts_reflection_neg_array[i],
                relations_3wm,
                relations_4wm,
                coeffs_3wm,
                coeffs_4wm,
            ),
            atol=1e-14,
            rtol=1e-10,
            max_num_steps=0,
            t_eval=x_array,
            rk_method=1,
            first_step=1,
        )
    return I_triplets


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

    # Determine number of modes from the first data element
    n_modes = len(data_kappas_gammas_array[0][0])  # kappas is always the first element

    # Convert relations_coefficients to numpy arrays with proper dtypes
    relations_3wm, relations_4wm, coeffs_3wm, coeffs_4wm = relations_coefficients
    relations_3wm = np.array(relations_3wm, dtype=np.int64)
    relations_4wm = np.array(relations_4wm, dtype=np.int64)
    coeffs_3wm = np.array(coeffs_3wm, dtype=np.complex128)  # Ensure complex type
    coeffs_4wm = np.array(coeffs_4wm, dtype=np.complex128)  # Ensure complex type

    # Convert list of heterogeneous data to properly typed arrays
    if with_loss and not reflections:
        # Loss-only model: data contains [kappas, alphas]
        kappas_array = np.empty((n_freq, n_modes), dtype=np.float64)
        alphas_array = np.empty((n_freq, n_modes), dtype=np.float64)

        for i in range(n_freq):
            kappas_array[i] = np.array(data_kappas_gammas_array[i][0], dtype=np.float64)
            alphas_array[i] = np.array(data_kappas_gammas_array[i][1], dtype=np.float64)

        return cme_general_solve_freq_array_loss(
            x_array,
            y0_array,
            kappas_array,
            alphas_array,
            relations_3wm,
            relations_4wm,
            coeffs_3wm,
            coeffs_4wm,
            thin,
        )

    elif reflections and not with_loss:
        # Full model: data contains [kappas, alphas, ts_reflection, ts_reflection_neg]
        kappas_array = np.empty((n_freq, n_modes), dtype=np.float64)
        alphas_array = np.empty((n_freq, n_modes), dtype=np.float64)
        ts_reflection_array = np.empty((n_freq, n_modes), dtype=np.complex128)
        ts_reflection_neg_array = np.empty((n_freq, n_modes), dtype=np.complex128)

        for i in range(n_freq):
            kappas_array[i] = np.array(data_kappas_gammas_array[i][0], dtype=np.float64)
            alphas_array[i] = np.array(data_kappas_gammas_array[i][1], dtype=np.float64)
            ts_reflection_array[i] = np.array(
                data_kappas_gammas_array[i][2], dtype=np.complex128
            )
            ts_reflection_neg_array[i] = np.array(
                data_kappas_gammas_array[i][3], dtype=np.complex128
            )

        return cme_general_solve_freq_array_full(
            x_array,
            y0_array,
            kappas_array,
            alphas_array,
            ts_reflection_array,
            ts_reflection_neg_array,
            relations_3wm,
            relations_4wm,
            coeffs_3wm,
            coeffs_4wm,
            thin,
        )

    else:
        # Ideal model: data contains only [kappas]
        kappas_array = np.empty((n_freq, n_modes), dtype=np.float64)

        for i in range(n_freq):
            kappas_array[i] = np.array(data_kappas_gammas_array[i][0], dtype=np.float64)

        return cme_general_solve_freq_array_ideal(
            x_array,
            y0_array,
            kappas_array,
            relations_3wm,
            relations_4wm,
            coeffs_3wm,
            coeffs_4wm,
            thin,
        )


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
        k = k_signal[i]
        _, I_triplets[i], _, _ = nbrk_ode(
            CMEode_complete,
            x_span,
            y0_broadcast[i],
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
            t_eval=x_array,
            rk_method=1,
            first_step=1,
        )
    return I_triplets
