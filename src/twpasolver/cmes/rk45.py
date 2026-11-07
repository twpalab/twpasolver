"""Cacheable RK45 integrator for use inside njit dispatch functions.

Adapted from CyRK's ``nbsolve_ivp`` (https://github.com/jrenaud90/CyRK,
Copyright Joe P. Renaud and CyRK contributors, Apache License 2.0), reduced
to the Dormand-Prince RK45 path without ``capture_extra``/``interpolate_extra``.

CyRK's ``nbsolve_ivp`` cannot be cached: its Butcher-tableau coefficients are
module-level ``np.array`` globals, which Numba compiles as "dynamic globals"
(runtime addresses that do not survive a restart), and that disables caching
of every njit caller, including the parallel dispatch functions in
base.py/bloch_fb.py (``NumbaWarning: Cannot cache ... as it uses dynamic
globals``). ``rk45_solve`` avoids this by building the tableau inside the
function body, and by being ``inline="always"`` with a statically known
``diffeq`` at each call site, so no runtime function pointer survives into
the caller's compiled code.
"""

from typing import Tuple

import numba as nb
import numpy as np

EPS = float(np.finfo(np.float64).eps)

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
    Adaptive Dormand-Prince RK45.

    Scalar tolerances, a given positive ``first_step`` and a non-empty
    ``t_eval`` sorted along the integration direction (the only way it is
    called); returns the solution on ``t_eval`` from the fourth-order dense
    output, shape ``(y_size, t_eval.size)``, NaN past a failed integration.
    """
    # Elementwise assignment: an np.array([...]) literal inside an inline="always"
    # function under parallel=True hits a Numba compiler bug.
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

    # Fourth-order continuous extension of Dormand-Prince (as in SciPy's RK45):
    # y(t_old + x h) = y_old + h sum_j K_j sum_p P[j, p] x^(p+1).
    P = np.zeros((7, 4), dtype=np.float64)
    P[0, 0] = 1.0
    P[0, 1] = -8048581381 / 2820520608
    P[0, 2] = 8663915743 / 2820520608
    P[0, 3] = -12715105075 / 11282082432
    P[2, 1] = 131558114200 / 32700410799
    P[2, 2] = -68118460800 / 10900136933
    P[2, 3] = 87487479700 / 32700410799
    P[3, 1] = -1754552775 / 470086768
    P[3, 2] = 14199869525 / 1410260304
    P[3, 3] = -10690763975 / 1880347072
    P[4, 1] = 127303824393 / 49829197408
    P[4, 2] = -318862633887 / 49829197408
    P[4, 3] = 701980252875 / 199316789632
    P[5, 1] = -282668133 / 205662961
    P[5, 2] = 2019193451 / 616988883
    P[5, 3] = -1453857185 / 822651844
    P[6, 1] = 40617522 / 29380423
    P[6, 2] = -110615467 / 29380423
    P[6, 3] = 69997945 / 29380423

    t_start = t_span[0]
    t_end = t_span[1]
    direction = np.sign(t_end - t_start) if t_end != t_start else 1
    y0 = np.asarray(y0)
    y_size = y0.size
    y_size_sqrt = np.sqrt(y_size)
    dtype = y0.dtype

    # Solution on t_eval (sorted along the integration direction), filled by
    # the dense output of every accepted step; unreached points stay NaN.
    t_eval = np.asarray(t_eval, dtype=np.float64)
    n_eval = t_eval.size
    y_results_reduced = np.full((y_size, n_eval), np.nan, dtype=dtype)
    w_dense = np.empty(7, dtype=np.float64)
    k_eval = 0
    while k_eval < n_eval and direction * (t_eval[k_eval] - t_start) <= 0:
        for i in range(y_size):
            y_results_reduced[i, k_eval] = y0[i]
        k_eval += 1

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

    # Envelopes vary slowly on the cell scale, so more steps than cells means a
    # runaway solution: stop with status -7, unreached points stay NaN.
    if max_steps_per_cell > 0:
        span_steps = int(np.ceil(np.abs(t_end - t_start) * max_steps_per_cell)) + 1
        if max_num_steps <= 0 or max_num_steps > span_steps:
            max_num_steps = span_steps
    use_max_steps = max_num_steps > 0

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

        # Dense output on the t_eval points of [t_old, t_new]; K[6] = f(t_new).
        while k_eval < n_eval and direction * (t_eval[k_eval] - t_new) <= 0:
            x = (t_eval[k_eval] - t_old) / step
            for j in range(rk_n_stages_plus1):
                w_dense[j] = x * (P[j, 0] + x * (P[j, 1] + x * (P[j, 2] + x * P[j, 3])))
            for i in range(y_size):
                acc = K[0, i] * w_dense[0]
                for j in range(2, rk_n_stages_plus1):
                    acc += K[j, i] * w_dense[j]
                y_results_reduced[i, k_eval] = y_old[i] + step * acc
            k_eval += 1

        t_old = t_new
        for i in range(y_size):
            y_old[i] = y_new[i]
            dydt_old[i] = dydt_new[i]
        len_t += 1

    return y_results_reduced
