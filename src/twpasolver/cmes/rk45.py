"""Cacheable RK45 integrator for use inside njit dispatch functions.

Adapted from CyRK's ``nbsolve_ivp`` (https://github.com/jrenaud90/CyRK,
Copyright Joe P. Renaud and CyRK contributors, Apache License 2.0), trimmed
to the Dormand-Prince RK45 path only, the only method this package ever
used, with the RK23/DOP853 branches and the
``capture_extra``/``interpolate_extra`` machinery (unused here) removed.

Why this exists, instead of just calling CyRK directly: CyRK's own
``nbsolve_ivp`` ships with ``cache=False`` because its Butcher-tableau
coefficients are module-level ``np.array`` globals, which Numba compiles as
"dynamic globals" (baked-in runtime addresses that don't survive a process
restart), and that taints caching for any njit function that calls it,
including every parallel dispatch function in base.py/bloch_fb.py, even
when they request ``cache=True`` themselves (confirmed via
``NumbaWarning: Cannot cache ... as it uses dynamic globals``).

``rk45_solve`` below avoids this two ways:
  - The Butcher-tableau arrays are built locally, inside the function body,
    instead of referenced from module scope, so there's no fixed address
    to bake in, just ordinary (cacheable) constant-folded code.
  - It is marked ``inline="always"`` and always called with a statically
    known ``diffeq`` at each call site, so after inlining no runtime
    function pointer survives into the caller's compiled code either.
"""

from typing import Tuple

import numba as nb
import numpy as np

from twpasolver.cmes.solver_config import SOLVER_MAX_STEPS_PER_CELL

EPS = np.finfo(np.float64).eps
EMPTY_ARR = np.empty(0, dtype=np.float64)

# Multiply steps computed from asymptotic behaviour of errors by this.
SAFETY = 0.9
MIN_FACTOR = 0.2  # Minimum allowed decrease in a step size.
MAX_FACTOR = 10.0  # Maximum allowed increase in a step size.

# Dormand-Prince RK45 Butcher tableau + embedded error coefficients.
RK45_ERROR_ESTIMATOR_ORDER = 4
RK45_N_STAGES = 6


@nb.njit(inline="always")
def rk45_solve(
    diffeq,
    t_span: Tuple[float, float],
    y0: np.ndarray,
    args: tuple = (),
    rtol: float = 1.0e-3,
    atol: float = 1.0e-6,
    rtols: np.ndarray = EMPTY_ARR,
    atols: np.ndarray = EMPTY_ARR,
    max_step: float = np.inf,
    first_step: float = None,
    t_eval: np.ndarray = EMPTY_ARR,
    max_num_steps: int = 0,
) -> np.ndarray:
    """
    Adaptive Dormand-Prince RK45, ported from CyRK's ``nbsolve_ivp``.

    Same calling convention as CyRK's ``nbsolve_ivp`` minus ``rk_method``
    (always RK45) and the extra-output options, and returns the solution
    array directly (shape ``(y_size, t_eval.size)``) instead of a result
    object, since callers here only ever used ``.y``.
    """
    # Built via elementwise assignment into a pre-allocated array rather than
    # np.array([...]) literal-list construction: under parallel=True, that
    # literal-list form triggers Numba's constant-array-inlining rewrite
    # pass, which hits an internal compiler bug for arrays built inside an
    # inline="always" function ("AssertionError: key already in dictionary:
    # '$np_typ_var.N'"). Plain elementwise assignment sidesteps that pass
    # entirely.
    C = np.empty(6, dtype=np.float64)
    C[0] = 0.0
    C[1] = 1 / 5
    C[2] = 3 / 10
    C[3] = 4 / 5
    C[4] = 8 / 9
    C[5] = 1.0

    A = np.zeros((6, 5), dtype=np.float64)
    A[1, 0] = 1 / 5
    A[2, 0] = 3 / 40
    A[2, 1] = 9 / 40
    A[3, 0] = 44 / 45
    A[3, 1] = -56 / 15
    A[3, 2] = 32 / 9
    A[4, 0] = 19372 / 6561
    A[4, 1] = -25360 / 2187
    A[4, 2] = 64448 / 6561
    A[4, 3] = -212 / 729
    A[5, 0] = 9017 / 3168
    A[5, 1] = -355 / 33
    A[5, 2] = 46732 / 5247
    A[5, 3] = 49 / 176
    A[5, 4] = -5103 / 18656

    B = np.empty(6, dtype=np.float64)
    B[0] = 35 / 384
    B[1] = 0.0
    B[2] = 500 / 1113
    B[3] = 125 / 192
    B[4] = -2187 / 6784
    B[5] = 11 / 84

    E = np.empty(7, dtype=np.float64)
    E[0] = -71 / 57600
    E[1] = 0.0
    E[2] = 71 / 16695
    E[3] = -71 / 1920
    E[4] = 17253 / 339200
    E[5] = -22 / 525
    E[6] = 1 / 40

    # Clean up and interpret inputs
    t_start = t_span[0]
    t_end = t_span[1]
    direction = np.sign(t_end - t_start) if t_end != t_start else 1
    y0 = np.asarray(y0)
    y_size = y0.size
    y_size_sqrt = np.sqrt(y_size)
    dtype = y0.dtype
    t_eval_size = t_eval.size

    y_result_store = np.empty(y_size, dtype=dtype)

    # Containers to store results during integration
    time_domain_list = [t_start]
    y_result_list = [np.copy(y0)]

    # Integrator status: 0 = ongoing, 1 = success, <0 = error.
    status = 0

    rk_n_stages = RK45_N_STAGES
    rk_n_stages_plus1 = rk_n_stages + 1
    A = np.asarray(A, dtype=dtype)
    B = np.asarray(B, dtype=dtype)
    E = np.asarray(E, dtype=dtype)
    K = np.empty((rk_n_stages_plus1, y_size), dtype=dtype)

    len_C = len(C)
    A_at_10 = A[1, 0]
    error_expo = 1.0 / (RK45_ERROR_ESTIMATOR_ORDER + 1.0)

    # Setup tolerances. rtols/atols, when non-empty, are assumed to already
    # be sized to y0 by the caller (validation is intentionally not done
    # here: raising inside this function, once inlined into the prange
    # loop it's always called from, gives the loop body more than one exit
    # point, which silently defeats Numba's parallelization of that loop).
    rtol_array = np.empty(y_size, dtype=np.float64)
    rtol_size = rtols.size
    if rtol_size > 0:
        for i in range(y_size):
            rtol_ = rtols[i]
            if rtol_ < (100 * EPS):
                rtol_ = 100 * EPS
            rtol_array[i] = rtol_
    else:
        if rtol < (100 * EPS):
            rtol = 100 * EPS
        for i in range(y_size):
            rtol_array[i] = rtol

    atol_array = np.empty(y_size, dtype=np.float64)
    atol_size = atols.size
    if atol_size > 0:
        for i in range(y_size):
            atol_array[i] = atols[i]
    else:
        for i in range(y_size):
            atol_array[i] = atol

    # Cap the number of accepted steps at the integration length in unit cells
    # (times SOLVER_MAX_STEPS_PER_CELL): the CMEs describe envelopes that vary
    # slowly on the scale of one cell, so needing more steps than cells means
    # the solution is running away (e.g. an unguarded stopband-edge process).
    # The run then stops with status -7 and the unreached points are NaN.
    if SOLVER_MAX_STEPS_PER_CELL > 0:
        span_steps = (
            int(np.ceil(np.abs(t_end - t_start) * SOLVER_MAX_STEPS_PER_CELL)) + 1
        )
        if max_num_steps <= 0 or max_num_steps > span_steps:
            max_num_steps = span_steps
    use_max_steps = max_num_steps > 0

    # Initialize variables for start of integration
    t_old = t_start
    t_new = t_start
    y_new = np.empty_like(y0)
    y_old = np.empty_like(y0)
    dydt_old = np.empty_like(y0)
    dydt_new = np.empty_like(y0)
    for i in range(y_size):
        y0_i = y0[i]
        y_new[i] = y0_i
        y_old[i] = y0_i

    output = np.asarray(diffeq(t_new, y_new, *args), dtype=dtype)
    for i in range(y_size):
        dydt_old[i] = output[i]

    # Find first step size. first_step, when given, is assumed valid (positive,
    # within [t_start, t_end]). See the note above about not raising here.
    first_step_found = False
    if first_step is not None:
        step_size = max_step
        if first_step != 0.0:
            step_size = first_step
            first_step_found = True

    if not first_step_found:
        # Select an initial step size based on the differential equation.
        # .. [1] E. Hairer, S. P. Norsett G. Wanner, "Solving Ordinary Differential
        #        Equations I: Nonstiff Problems", Sec. II.4.
        if y_size == 0:
            step_size = np.inf
        else:
            d0 = 0.0
            d1 = 0.0
            for i in range(y_size):
                scale = atol_array[i] + np.abs(y_old[i]) * rtol_array[i]
                d0_abs = np.abs(y_old[i] / scale)
                d1_abs = np.abs(dydt_old[i] / scale)
                d0 += d0_abs * d0_abs
                d1 += d1_abs * d1_abs

            d0 = np.sqrt(d0) / y_size_sqrt
            d1 = np.sqrt(d1) / y_size_sqrt

            if d0 < 1.0e-5 or d1 < 1.0e-5:
                h0 = 1.0e-6
            else:
                h0 = 0.01 * d0 / d1

            y1 = y_old + h0 * direction * dydt_old
            t1 = t_old + h0 * direction

            diffeq_output = np.asarray(diffeq(t1, y1, *args), dtype=dtype)
            for i in range(y_size):
                dydt_new[i] = diffeq_output[i]

            d2 = 0.0
            for i in range(y_size):
                scale = atol_array[i] + np.abs(y_old[i]) * rtol_array[i]
                d2_abs = np.abs((dydt_new[i] - dydt_old[i]) / scale)
                d2 += d2_abs * d2_abs
            d2 = np.sqrt(d2) / (h0 * y_size_sqrt)

            if d1 <= 1.0e-15 and d2 <= 1.0e-15:
                h1 = max(1.0e-6, h0 * 1.0e-3)
            else:
                h1 = (0.01 / max(d1, d2)) ** error_expo

            next_after = 10.0 * abs(np.nextafter(t_old, direction * np.inf) - t_old)
            step_size = max(next_after, min(100.0 * h0, h1))

    if y_size == 0:
        status = -6

    len_t = 1  # Already one time step due to initial conditions.
    while status == 0:
        if t_new == t_end:
            t_old = t_end
            t_new = t_end
            status = 1
            break

        if use_max_steps and len_t > max_num_steps:
            status = -7
            break

        next_after = 10.0 * abs(np.nextafter(t_old, direction * np.inf) - t_old)
        min_step = next_after

        if step_size > max_step:
            step_size = max_step
        elif step_size < min_step:
            step_size = min_step

        step_accepted = False
        step_rejected = False
        step_error = False
        while not step_accepted:
            if step_size < min_step:
                step_error = True
                break

            step = step_size * direction
            t_new = t_old + step

            if direction * (t_new - t_end) > 0:
                t_new = t_end
                step = t_new - t_old
                step_size = np.abs(step)

            for s in range(1, len_C):
                time_ = t_old + C[s] * step
                if s == 1:
                    for i in range(y_size):
                        temp_double_numeric = dydt_old[i]
                        K[0, i] = temp_double_numeric
                        y_new[i] = y_old[i] + (temp_double_numeric * A_at_10 * step)
                else:
                    for j in range(s):
                        A_at_sj = A[s, j] * step
                        for i in range(y_size):
                            if j == 0:
                                y_new[i] = y_old[i]
                            y_new[i] += K[j, i] * A_at_sj

                diffeq_output = np.asarray(diffeq(time_, y_new, *args), dtype=dtype)
                for i in range(y_size):
                    K[s, i] = diffeq_output[i]

            for j in range(rk_n_stages):
                temp = B[j] * step
                for i in range(y_size):
                    if j == 0:
                        y_new[i] = y_old[i]
                    y_new[i] += K[j, i] * temp

            diffeq_output = np.asarray(diffeq(t_new, y_new, *args), dtype=dtype)
            for i in range(y_size):
                dydt_new[i] = diffeq_output[i]

            error_norm = 0.0
            for i in range(y_size):
                scale = (
                    atol_array[i]
                    + max(np.abs(y_old[i]), np.abs(y_new[i])) * rtol_array[i]
                )
                K[rk_n_stages, i] = dydt_new[i]
                error_dot_1 = 0.0
                for j in range(rk_n_stages_plus1):
                    error_dot_1 += K[j, i] * E[j]

                error_norm_abs = np.abs(error_dot_1)
                error_norm_abs *= step / scale
                error_norm += error_norm_abs * error_norm_abs
            error_norm = np.sqrt(error_norm) / y_size_sqrt

            if error_norm < 1.0:
                if error_norm == 0.0:
                    step_factor = MAX_FACTOR
                else:
                    step_factor = min(MAX_FACTOR, SAFETY * error_norm**-error_expo)

                if step_rejected:
                    step_factor = min(step_factor, 1.0)

                step_size = step_size * step_factor
                step_accepted = True
            else:
                step_size = step_size * max(
                    MIN_FACTOR, SAFETY * error_norm**-error_expo
                )
                step_rejected = True

        if step_error:
            status = -1
            break
        elif not step_accepted:
            status = -7
            break

        t_old = t_new
        for i in range(y_size):
            y_old[i] = y_new[i]
            dydt_old[i] = dydt_new[i]
            y_result_store[i] = y_new[i]

        time_domain_list.append(t_new)
        y_result_list.append(np.copy(y_result_store))
        len_t += 1

    # To match the format that scipy follows, we will take the transpose of y.
    time_domain = np.empty(len_t, dtype=np.float64)
    y_results = np.empty((y_size, len_t), dtype=dtype)
    for t_i in range(len_t):
        time_domain[t_i] = time_domain_list[t_i]
        y_results_list_at_t = y_result_list[t_i]
        for y_i in range(y_size):
            y_results[y_i, t_i] = y_results_list_at_t[y_i]

    if t_eval_size == 0:
        return y_results

    # No dense output implemented (matching CyRK's own approach): linearly
    # interpolate the accepted-step trajectory onto the requested grid.
    t_eval = np.asarray(t_eval, dtype=np.float64)
    y_results_reduced = np.empty((y_size, t_eval.size), dtype=dtype)
    for i in range(y_size):
        y_results_reduced[i, :] = np.interp(t_eval, time_domain, y_results[i, :])
    if status != 1:
        # Failed integration: do not extrapolate the last accepted point.
        t_reached = time_domain[len_t - 1]
        for k in range(t_eval.size):
            if direction * (t_eval[k] - t_reached) > 0:
                for i in range(y_size):
                    y_results_reduced[i, k] = np.nan

    return y_results_reduced
