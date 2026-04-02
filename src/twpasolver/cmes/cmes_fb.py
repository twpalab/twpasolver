"""
Coupled Mode Equations — forward/backward solver with inlined RK45.

The RK45 integrator is inlined directly into the per-frequency solve
function rather than passed as a higher-order argument.  This avoids
Numba's parallel-compilation deadlock that occurs when a function
pointer is specialised inside a prange body.

The single RHS function (cme_rhs) is called by name from the inlined
solver, giving Numba a fully static call graph throughout.
"""

import numba as nb
import numpy as np

from twpasolver.bonus_types import (
    ComplexArray,
    FloatArray,
    nb_complex1d,
    nb_complex2d,
    nb_float1d,
    nb_int2d,
)
from twpasolver.cmes.solver_config import *
from CyRK import nbsolve_ivp

@nb.njit(cache=True)
def _filter_wm_terms(kappas, relations_3wm, relations_4wm, relative_deltak_cutoff):
    num_modes = len(kappas)
    valid_3wm = np.empty((len(relations_3wm) * 4, 3), dtype=relations_3wm.dtype)
    valid_4wm = np.empty((len(relations_4wm) * 8, 4), dtype=relations_4wm.dtype)
    nv3 = 0
    nv4 = 0
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

    for k3 in range(len(relations_3wm)):
        r = relations_3wm[k3].copy()
        bk = kappas[r[0]]
        m1, m2 = r[1], r[2]
        c1 = m1 >= num_modes
        c2 = m2 >= num_modes
        k1 = m1 - num_modes if c1 else m1
        k2 = m2 - num_modes if c2 else m2
        for si in range(4):
            s1, s2 = signs_3wm[si]
            dk = (
                -bk
                + s1 * kappas[k1] * (-1 if c1 else 1)
                + s2 * kappas[k2] * (-1 if c2 else 1)
            )
            if np.abs(dk / bk) < relative_deltak_cutoff:
                valid_3wm[nv3, 0] = r[0]
                valid_3wm[nv3, 1] = r[1] + (2 * num_modes if s1 == -1 else 0)
                valid_3wm[nv3, 2] = r[2] + (2 * num_modes if s2 == -1 else 0)
                nv3 += 1

    for k4 in range(len(relations_4wm)):
        r = relations_4wm[k4].copy()
        bk = kappas[r[0]]
        m1, m2, m3 = r[1], r[2], r[3]
        c1 = m1 >= num_modes
        c2 = m2 >= num_modes
        c3 = m3 >= num_modes
        k1 = m1 - num_modes if c1 else m1
        k2 = m2 - num_modes if c2 else m2
        k3 = m3 - num_modes if c3 else m3
        for si in range(8):
            s1, s2, s3 = signs_4wm[si]
            dk = (
                -bk
                + s1 * kappas[k1] * (-1 if c1 else 1)
                + s2 * kappas[k2] * (-1 if c2 else 1)
                + s3 * kappas[k3] * (-1 if c3 else 1)
            )
            if np.abs(dk / bk) < relative_deltak_cutoff:
                valid_4wm[nv4, 0] = r[0]
                valid_4wm[nv4, 1] = r[1] + (2 * num_modes if s1 == -1 else 0)
                valid_4wm[nv4, 2] = r[2] + (2 * num_modes if s2 == -1 else 0)
                valid_4wm[nv4, 3] = r[3] + (2 * num_modes if s3 == -1 else 0)
                nv4 += 1

    return valid_3wm[:nv3], valid_4wm[:nv4]


# ---------------------------------------------------------------------------
# CME right-hand side  (called by name from the inlined solver below)
# ---------------------------------------------------------------------------
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
        nb_float1d,
        nb_complex2d,
        nb.float64,
    ),
    cache=True,
)
def cme_rhs(
    x,
    currents,
    alphas,
    kappas,
    relations_3wm,
    relations_4wm,
    coeffs_3wm,
    coeffs_4wm,
    x_evals,
    currents_evals,
    end,
):
    """
    Unified CME RHS for forward and backward passes.

    currents_evals holds the reversed counter-propagating envelope B_m.
    When it is all zeros (first forward pass) the counter term vanishes,
    reproducing the no-backward-field equation exactly.

    Output length = 2*n_modes:
      [0:n]   dB_m/dx  (full, with +alpha*B term)
      [n:2n]  raw nonlinear part only (for reflection coefficient use)
    """
    n = len(currents)
    out = np.zeros(2 * n, nb.complex128)
    ep = np.exp(1j * kappas * x)
    en = np.exp(1j * kappas * (end - x))

    ar = np.empty(2 * n, dtype=nb.complex128)
    for i in range(n):
        ctr = (
            np.interp(x, x_evals, currents_evals[i].real)
            + 1j * np.interp(x, x_evals, currents_evals[i].imag)
        ) * en[i]
        ar[i] = currents[i] * ep[i] + ctr
        ar[i + n] = np.conj(ar[i])

    for idx, i1, i2 in relations_3wm:
        out[idx] += coeffs_3wm[idx] * ar[i1] * ar[i2]
    for idx, i1, i2, i3 in relations_4wm:
        out[idx] += coeffs_4wm[idx] * ar[i1] * ar[i2] * ar[i3]

    for i in range(n):
        raw = (alphas[i] + 1j * kappas[i]) * out[i] / ep[i]
        out[i] = raw + alphas[i] * currents[i]
        out[n + i] = raw

    return out


# ---------------------------------------------------------------------------
# Parallel outer loop
# ---------------------------------------------------------------------------
@nb.njit(parallel=True)
def general_cmes_solve_fb_cutoff(
    x_array: FloatArray,
    y0_array_fwd: ComplexArray,
    y0_array_bwd: ComplexArray,
    gammas_array: ComplexArray,
    reflections_array: ComplexArray,
    relations_3wm,
    relations_4wm,
    coeffs_3wm,
    coeffs_4wm,
    passes: int = 3,
    kappa_cutoff: float = 0.1,
    update_rate: float = 0.8,
    convergence_threshold: float = 0.05,
    save_all_passes: bool = False,
    Z0: float = 50,
) -> ComplexArray:
    """
    Solve full general CMEs (forward + backward) for all frequency points.

    The parallel body contains only calls to statically-known njit functions
    (_filter_wm_terms, _solve_one) with no function-pointer arguments, giving
    Numba a fully static call graph and avoiding the parallel-compilation
    deadlock that occurs with higher-order function arguments in prange.
    """
    n_freq = gammas_array.shape[0]
    n_modes = gammas_array.shape[1]
    n_out = 2 * n_modes
    n_x = len(x_array)
    x_end = x_array[-1]
    x_span_0 = x_array[0]
    x_span = (x_array[0], x_array[-1])
    trans_in = 1 - reflections_array
    Zb = Z0 * (1 + reflections_array) / (1 - reflections_array)

    if save_all_passes:
        I_out = np.empty((2 * n_freq * passes, n_modes, n_x), dtype=np.complex128)
    else:
        I_out = np.empty((2 * n_freq, n_modes, n_x), dtype=np.complex128)

    # ---- thread-local buffers (never aliased to solver internals) ---- #
    A_fwd = np.zeros((n_freq, n_out, n_x), dtype=np.complex128)
    A_bwd = np.zeros((n_freq, n_out, n_x), dtype=np.complex128)
    A_tmp = np.zeros((n_freq, n_out, n_x), dtype=np.complex128)
    # Counter-propagating field passed to cme_rhs.
    # Initialised to zero so the first forward pass has no backward source.
    A_counter = np.zeros((n_freq, n_modes, n_x), dtype=np.complex128)

    for i in nb.prange(n_freq):
        I_fwd = np.zeros(n_modes, dtype=np.complex128)
        I_bwd = np.zeros(n_modes, dtype=np.complex128)
        refl = np.zeros(n_modes, dtype=np.complex128)

        converged = False
        stable_count = 0
        prev_fwd = np.zeros(n_modes, dtype=np.complex128)
        prev_bwd = np.zeros(n_modes, dtype=np.complex128)

        alphas_i = np.ascontiguousarray(gammas_array[i].real)
        kappas_i = np.ascontiguousarray(gammas_array[i].imag)
        refl_i = np.ascontiguousarray(reflections_array[i])
        trans_i = np.ascontiguousarray(trans_in[i])
        Zb_i = np.ascontiguousarray(Zb[i])
        x_array_i = np.ascontiguousarray(x_array)

        for j in range(n_modes):
            I_fwd[j] = y0_array_fwd[i, j] * trans_i[j]
            I_bwd[j] = y0_array_bwd[i, j] * trans_i[j]
            refl[j] = refl_i[j]

       # r3, r4 = _filter_wm_terms(kappas_i, relations_3wm, relations_4wm, kappa_cutoff)

        for k in range(passes):
            A_fwd[i] = nbsolve_ivp(
                cme_rhs,
                x_span,
                I_fwd,
                args=(
                    alphas_i,
                    kappas_i,
                    relations_3wm,
                    relations_4wm,
                    coeffs_3wm,
                    coeffs_4wm,
                    x_array_i,
                    A_counter[i],
                    x_end,
                ),
                atol=SOLVER_ATOL,
                rtol=SOLVER_RTOL,
                max_num_steps=x_end,
                t_eval=x_array_i,
                rk_method=SOLVER_RK_METHOD,
                first_step=SOLVER_FIRST_STEP,
                capture_extra=True
            ).y

            # ---- reflection at far end → backward IC ----------------- #
            for j in range(n_modes):
                A_fwd_end_j = A_fwd[i, j, -1] * np.exp(1j * kappas_i[j] * x_end)
                I_bwd[j] = y0_array_bwd[i, j] * trans_i[j] + refl[j] * A_fwd_end_j

            # ---- backward solve --------------------------------------- #
            # Reverse A_fwd i, into counter buffer.
            for j in range(n_modes):
                for ix in range(n_x):
                    A_counter[i, j, ix] = A_fwd[i, j, n_x - 1 - ix]

            A_bwd[i] = nbsolve_ivp(
                cme_rhs,
                x_span,
                I_bwd,
                args=(
                    alphas_i,
                    kappas_i,
                    relations_3wm,
                    relations_4wm,
                    coeffs_3wm,
                    coeffs_4wm,
                    x_array_i,
                    A_counter[i],
                    x_end,
                ),
                atol=SOLVER_ATOL,
                rtol=SOLVER_RTOL,
                max_num_steps=x_end,
                t_eval=x_array_i,
                rk_method=SOLVER_RK_METHOD,
                first_step=SOLVER_FIRST_STEP,
                capture_extra=True
            ).y

            # # ---- update refl (fwd end → next bwd IC) ----------------- #
            if k > 0:
                for j in range(n_modes):
                    de        = np.exp(-alphas_i[j] * x_end)
                    raw_bwd   = A_bwd[i, n_modes+j, 0]
                    B_bwd0    = A_bwd[i, j, 0]
                    raw_fwd   = A_fwd[i, n_modes+j, -1] * de
                    B_fwd_end = A_fwd[i, j, -1]
                    gj        = gammas_array[i, j]
                    Zbj       = Zb_i[j]
                    refl[j]   = (
                        -B_fwd_end / B_bwd0
                        * ((Z0-Zbj)*gj*B_bwd0 + Zbj*raw_bwd)
                        / ((Z0+Zbj)*gj*B_fwd_end + Zbj*raw_fwd)
                   )

            # ---- reverse A_bwd → counter; update fwd IC -------------- #
            for j in range(n_modes):
                for ix in range(n_x):
                    A_counter[i, j, ix] = A_bwd[i, j, n_x - 1 - ix]

            if k > 0:
                for j in range(n_modes):
                    de        = np.exp(-alphas_i[j] * x_end)
                    raw_fwd   = A_fwd[i, n_modes+j, 0]
                    B_fwd0    = A_fwd[i, j, 0]
                    raw_bwd   = A_bwd[i, n_modes+j, -1]
                    B_bwd_end = A_bwd[i, j, -1] * de
                    gj        = gammas_array[i, j]
                    Zbj       = Zb_i[j]
                    rb        = (
                        -B_bwd_end / B_fwd0
                        * ((Z0-Zbj)*gj*B_fwd0 + Zbj*raw_fwd)
                        / ((Z0+Zbj)*gj*B_bwd_end + Zbj*raw_bwd)
                    )
                    A_bwd_end_j = A_bwd[i, j,-1] * np.exp(1j*kappas_i[j]*x_end)
                    I_fwd[j] = y0_array_fwd[i, j]*trans_i[j] + rb*A_bwd_end_j
            else:
                for j in range(n_modes):
                    A_bwd_end_j = A_bwd[i, j, -1] * np.exp(1j * kappas_i[j] * x_end)
                    I_fwd[j] = y0_array_fwd[i, j] * trans_i[j] + refl_i[j] * A_bwd_end_j

            # # ---- convergence check ----------------------------------- #
            if k > 1:
                nf = 0.0
                nb_ = 0.0
                for j in range(n_modes):
                    v = np.abs(A_fwd[i, j,-1] - prev_fwd[j]) / (np.abs(A_fwd[i, j,-1]) + 1e-30)
                    w = np.abs(A_bwd[i, j,-1] - prev_bwd[j]) / (np.abs(A_bwd[i, j,-1]) + 1e-30)
                    if v > nf: nf = v
                    if w > nb_: nb_ = w
                if nf < convergence_threshold and nb_ < convergence_threshold:
                    stable_count += 1
                else:
                    stable_count = 0
                if stable_count >= 4:
                    converged = True

            # ---- store output ------------------------------------------- #
            if save_all_passes:
                I_out[i + k * n_freq] = A_fwd[i, :n_modes, :]
                I_out[i + k * n_freq + n_freq * passes] = A_bwd[i, :n_modes, :]
            else:
                I_out[i] = A_fwd[i, :n_modes, :]
                I_out[i + n_freq] = A_bwd[i, :n_modes, :]

            for j in range(n_modes):
                prev_fwd[j] = A_fwd[i, j, -1]
                prev_bwd[j] = A_bwd[i, j, -1]

    return I_out
