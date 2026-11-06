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

EPS = np.finfo(np.float64).eps

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
    t_eval: np.ndarray,
    first_step: float,
    args: tuple = (),
    rtol: float = 1.0e-3,
    atol: float = 1.0e-6,
    max_num_steps: int = 0,
    max_steps_per_cell: float = 0.0,
) -> np.ndarray:
    """
    Adaptive Dormand-Prince RK45, ported from CyRK's ``nbsolve_ivp``.

    Scalar tolerances, a given positive ``first_step`` and a non-empty
    ``t_eval`` (the only way it is called); returns the solution interpolated
    on ``t_eval``, shape ``(y_size, t_eval.size)``.
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

    if rtol < (100 * EPS):
        rtol = 100 * EPS

    # Cap the number of accepted steps at the integration length in unit cells
    # (times max_steps_per_cell): the CMEs describe envelopes that vary
    # slowly on the scale of one cell, so needing more steps than cells means
    # the solution is running away (e.g. an unguarded stopband-edge process).
    # The run then stops with status -7 and the unreached points are NaN.
    if max_steps_per_cell > 0:
        span_steps = int(np.ceil(np.abs(t_end - t_start) * max_steps_per_cell)) + 1
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

    step_size = first_step

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

        if step_size < min_step:
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
                scale = atol + max(np.abs(y_old[i]), np.abs(y_new[i])) * rtol
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
