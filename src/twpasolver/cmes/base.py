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
    nb_float2d,
    nb_int2d,
)
from twpasolver.cmes.solver_config import (
    SOLVER_ATOL,
    SOLVER_FIRST_STEP,
    SOLVER_MAX_STEPS,
    SOLVER_RK_METHOD,
    SOLVER_RTOL,
)


@nb.njit(
    nb.types.UniTuple(nb.int64, 2)(nb_float1d, nb.float64),
    cache=True,
    inline="always",
)
def _sorted_interp_bracket(xp: nb_float1d, x: nb.float64):
    """
    Return (lo, hi) bracket indices for x in the sorted grid xp.

    Callers compute frac = (x - xp[lo]) / (xp[hi] - xp[lo]) and apply it
    to as many value arrays as needed, paying for the binary search only once.

    Boundary behaviour (matching np.interp):
      x <= xp[0]       → lo = hi = 0        (frac = 0 → clamp to fp[0])
      x >= xp[n-1]     → lo = hi = n-1      (frac = 0 → clamp to fp[n-1])
      interior          → lo = hi - 1,  hi = searchsorted result
    """
    n = len(xp)
    if x <= xp[0]:
        return nb.int64(0), nb.int64(0)
    if x >= xp[n - 1]:
        return nb.int64(n - 1), nb.int64(n - 1)
    hi = nb.int64(np.searchsorted(xp, x))  # xp[hi-1] < x <= xp[hi]
    return hi - nb.int64(1), hi


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
    """RHS of the no-reflections CME system for a uniform device."""
    num_modes = len(currents)
    derivs = np.zeros(num_modes, nb.complex128)
    exp_pos = np.exp(1j * kappas * x)

    alphas_rhs = np.zeros(2 * num_modes, dtype=np.complex128)
    for i in range(num_modes):
        alphas_rhs[i] = currents[i] * exp_pos[i]
        alphas_rhs[i + num_modes] = np.conj(alphas_rhs[i])

    for i, (idx, idx1, idx2) in enumerate(relations_3wm):
        derivs[idx] += coeffs_3wm[i] * alphas_rhs[idx1] * alphas_rhs[idx2]
    for i, (idx, idx1, idx2, idx3) in enumerate(relations_4wm):
        derivs[idx] += (
            coeffs_4wm[i] * alphas_rhs[idx1] * alphas_rhs[idx2] * alphas_rhs[idx3]
        )
    derivs = (alphas + 1j * kappas) * derivs / exp_pos + alphas * currents
    return derivs


@nb.njit(
    nb_complex1d(
        nb.float64,
        nb_complex1d,
        nb_float2d,
        nb_float2d,
        nb_int2d,
        nb_int2d,
        nb_complex2d,
        nb_complex2d,
        nb_float1d,
        nb_float1d,
        nb_float2d,
    ),
    cache=True,
)
def no_reflections_cmes_segmented(
    x,
    currents,
    alphas_table,
    kappas_table,
    relations_3wm,
    relations_4wm,
    coeffs_3wm_table,
    coeffs_4wm_table,
    segment_midpoints,
    x_array,
    phi_table,
):
    """
    RHS of the no-reflections CME system for a segmented (position-dependent) device.

    Parameters
    ----------
    alphas_table, kappas_table : ndarray, shape (n_modes, n_segments)
        Local attenuation and wavenumber, interpolated at ``x`` via
        ``segment_midpoints``.
    coeffs_3wm_table, coeffs_4wm_table : ndarray, shape (n_relations, n_segments)
        Position-dependent nonlinear coupling coefficients, on the same grid.
    phi_table : ndarray, shape (n_modes, n_x)
        Cumulative phase per mode, precomputed on the solver output grid
        ``x_array``.
    """
    num_modes = len(currents)
    derivs = np.zeros(num_modes, nb.complex128)

    # Locate x in both sorted grids via one binary search each.
    lo, hi = _sorted_interp_bracket(segment_midpoints, x)
    lo_x, hi_x = _sorted_interp_bracket(x_array, x)

    frac = (
        0.0
        if lo == hi
        else (x - segment_midpoints[lo])
        / (segment_midpoints[hi] - segment_midpoints[lo])
    )
    frac_x = (
        0.0 if lo_x == hi_x else (x - x_array[lo_x]) / (x_array[hi_x] - x_array[lo_x])
    )

    # Local alphas (via linear interp) and exact cumulative phase (via phi_table).
    alphas = np.empty(num_modes, dtype=nb.float64)
    kappas = np.empty(num_modes, dtype=nb.float64)
    exp_pos = np.empty(num_modes, dtype=nb.complex128)
    for i in range(num_modes):
        alphas[i] = alphas_table[i, lo] + frac * (
            alphas_table[i, hi] - alphas_table[i, lo]
        )
        kappas[i] = kappas_table[i, lo] + frac * (
            kappas_table[i, hi] - kappas_table[i, lo]
        )
        phi = phi_table[i, lo_x] + frac_x * (phi_table[i, hi_x] - phi_table[i, lo_x])
        exp_pos[i] = np.exp(1j * phi)

    # Local nonlinearity coefficients.
    n_rel3 = coeffs_3wm_table.shape[0]
    coeffs_3wm = np.empty(n_rel3, dtype=nb.complex128)
    for r in range(n_rel3):
        coeffs_3wm[r] = coeffs_3wm_table[r, lo] + frac * (
            coeffs_3wm_table[r, hi] - coeffs_3wm_table[r, lo]
        )

    n_rel4 = coeffs_4wm_table.shape[0]
    coeffs_4wm = np.empty(n_rel4, dtype=nb.complex128)
    for r in range(n_rel4):
        coeffs_4wm[r] = coeffs_4wm_table[r, lo] + frac * (
            coeffs_4wm_table[r, hi] - coeffs_4wm_table[r, lo]
        )

    alphas_rhs = np.zeros(2 * num_modes, dtype=np.complex128)
    for i in range(num_modes):
        alphas_rhs[i] = currents[i] * exp_pos[i]
        alphas_rhs[i + num_modes] = np.conj(alphas_rhs[i])

    for i, (idx, idx1, idx2) in enumerate(relations_3wm):
        derivs[idx] += coeffs_3wm[i] * alphas_rhs[idx1] * alphas_rhs[idx2]
    for i, (idx, idx1, idx2, idx3) in enumerate(relations_4wm):
        derivs[idx] += (
            coeffs_4wm[i] * alphas_rhs[idx1] * alphas_rhs[idx2] * alphas_rhs[idx3]
        )
    derivs = (alphas + 1j * kappas) * derivs / exp_pos + alphas * currents
    return derivs


@nb.njit(parallel=True)
def _no_reflections_solve_uniform(
    x_array,
    y0_broadcast,
    gammas_array,
    relations_3wm,
    relations_4wm,
    coeffs_3wm,
    coeffs_4wm,
):
    """Parallel solve for uniform device; gammas_array shape (n_freq, n_modes)."""
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


@nb.njit(parallel=True)
def _no_reflections_solve_segmented(
    x_array,
    y0_broadcast,
    gammas_array,
    relations_3wm,
    relations_4wm,
    coeffs_3wm_table,
    coeffs_4wm_table,
    segment_midpoints,
):
    """
    Parallel solve for segmented device; gammas_array shape (n_freq, n_modes, n_segments).

    For each frequency the cumulative phase Phi_m(x[k]) is precomputed on the
    output grid via the cumulative trapezoid rule applied to kappa_m linearly
    interpolated at x_array from segment_midpoints.  This table is then passed
    to the RHS so that _interp1d_sorted(x_array, phi_table[m], x) gives the exact
    phase at any integrator step x via a single binary search per mode per step.
    Both grids are sorted, so all interpolations use O(log n) binary search instead
    of O(n) linear scan.
    """
    n_freq = y0_broadcast.shape[0]
    n_modes = y0_broadcast.shape[1]
    n_x = len(x_array)
    x_span = (x_array[0], x_array[-1])
    I_tuples = np.empty((n_freq, n_modes, n_x), dtype=nb.complex128)

    for i in nb.prange(n_freq):
        # gammas_array[i]: (n_modes, n_segments)
        alphas_i = np.ascontiguousarray(gammas_array[i].real)
        kappas_i = np.ascontiguousarray(gammas_array[i].imag)

        # Precompute cumulative phase on the output grid: phi_table[m, k] = Phi_m(x[k]).
        # kappa_m(x) is piecewise-linear between segment_midpoints, so the
        # trapezoid rule on the interpolated values is exact for that interpolant.
        # Both x_array and segment_midpoints are sorted. For each pair (x[k-1], x[k])
        # we compute the segment bracket once and reuse it across all modes.
        phi_table = np.zeros((n_modes, n_x), dtype=nb.float64)
        for k in range(1, n_x):
            lo_l, hi_l = _sorted_interp_bracket(segment_midpoints, x_array[k - 1])
            fl = (
                0.0
                if lo_l == hi_l
                else (x_array[k - 1] - segment_midpoints[lo_l])
                / (segment_midpoints[hi_l] - segment_midpoints[lo_l])
            )
            lo_r, hi_r = _sorted_interp_bracket(segment_midpoints, x_array[k])
            fr = (
                0.0
                if lo_r == hi_r
                else (x_array[k] - segment_midpoints[lo_r])
                / (segment_midpoints[hi_r] - segment_midpoints[lo_r])
            )
            dx = x_array[k] - x_array[k - 1]
            for m in range(n_modes):
                kl = kappas_i[m, lo_l] + fl * (kappas_i[m, hi_l] - kappas_i[m, lo_l])
                kr = kappas_i[m, lo_r] + fr * (kappas_i[m, hi_r] - kappas_i[m, lo_r])
                phi_table[m, k] = phi_table[m, k - 1] + 0.5 * (kl + kr) * dx

        result = nbsolve_ivp(
            no_reflections_cmes_segmented,
            x_span,
            y0_broadcast[i],
            args=(
                alphas_i,
                kappas_i,
                relations_3wm,
                relations_4wm,
                coeffs_3wm_table,
                coeffs_4wm_table,
                segment_midpoints,
                x_array,
                phi_table,
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


def no_reflections_cmes_solve(
    x_array: FloatArray,
    y0_broadcast: ComplexArray,
    gammas_array: ComplexArray,
    relations_3wm,
    relations_4wm,
    coeffs_3wm,
    coeffs_4wm,
    segment_midpoints: FloatArray | None = None,
) -> ComplexArray:
    """
    Solve the no-reflections CME system for every signal frequency in parallel.

    Dispatches to the segmented or uniform Numba kernel based on
    ``gammas_array.ndim``:

    - Uniform: ``gammas_array`` shape (n_freq, n_modes), ``coeffs_*`` 1-D.
    - Segmented: ``gammas_array`` shape (n_freq, n_modes, n_seg),
      ``coeffs_*`` shape (n_rel, n_seg); ``segment_midpoints`` required.

    Returns
    -------
    I_tuples : (n_freq, n_modes, n_positions)
        Slow-envelope amplitudes (no propagation phase).  Apply
        ``_apply_phase_carrier`` in the caller to recover the full-phase field.
    """
    if gammas_array.ndim == 3:
        if segment_midpoints is None:
            raise ValueError(
                "segment_midpoints is required for segmented devices "
                "(gammas_array.ndim == 3)."
            )
        return _no_reflections_solve_segmented(
            np.ascontiguousarray(x_array),
            y0_broadcast,
            gammas_array,
            relations_3wm,
            relations_4wm,
            np.ascontiguousarray(coeffs_3wm),
            np.ascontiguousarray(coeffs_4wm),
            np.ascontiguousarray(segment_midpoints),
        )
    else:
        return _no_reflections_solve_uniform(
            x_array,
            y0_broadcast,
            gammas_array,
            relations_3wm,
            relations_4wm,
            coeffs_3wm,
            coeffs_4wm,
        )
