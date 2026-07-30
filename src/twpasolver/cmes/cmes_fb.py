"""Coupled Mode Equations forward/backward solver."""

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
from twpasolver.cmes.base import _sorted_interp_bracket
from twpasolver.cmes.solver_config import *


@nb.njit(cache=True)
def _compute_atol_vec(solution, n_modes, base_atol, scale=1e-2):
    """
    Compute a per-mode absolute tolerance vector from a completed pass.

    Parameters
    ----------
    solution  : (2*n_modes, n_x) complex128
        Raw solver output — rows 0..n_modes-1 are the slow envelopes.
    n_modes   : int
    base_atol : float
        Global floor (SOLVER_ATOL).
    scale     : float
        1e-2 → atol = 1% of minimum observed amplitude (~2 digits of headroom).

    Returns
    -------
    atol_vec : (n_modes,) float64
    """
    atol_vec = np.empty(n_modes, dtype=nb.float64)
    for m in range(n_modes):
        min_abs = np.inf
        for ix in range(solution.shape[1]):
            v = np.abs(solution[m, ix])
            if v < min_abs:
                min_abs = v
        if min_abs == 0.0 or min_abs == np.inf:
            atol_vec[m] = base_atol
        else:
            candidate = scale * min_abs
            atol_vec[m] = candidate if candidate > base_atol else base_atol
    return atol_vec


# Wave-mixing term filter


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
        r = np.ascontiguousarray(relations_3wm[k3])
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
        r = np.ascontiguousarray(relations_4wm[k4])
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


# Uniform RHS


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
    Unified CME RHS for uniform-device forward and backward passes.

    currents_evals holds the reversed counter-propagating slow envelope B_m.
    When it is all zeros (first forward pass) the counter term vanishes,
    reproducing the no-backward-field equation exactly.

    Phase carriers:
      forward  carrier: ep[m] = exp(i * kappas[m] * x)
      backward carrier: en[m] = exp(i * kappas[m] * (end - x))

    Output length = 2*n_modes:
      [0:n]   dB_m/dx  (full, with +alpha*B term)
      [n:2n]  raw nonlinear part only (for reflection-coefficient update)
    """
    n = len(currents)
    out = np.zeros(2 * n, nb.complex128)
    ep = np.exp(1j * kappas * x)
    en = np.exp(1j * kappas * (end - x))

    ar = np.empty(4 * n, dtype=nb.complex128)
    # x_evals is sorted: one binary search gives the bracket for all modes.
    lo_e, hi_e = _sorted_interp_bracket(x_evals, x)
    frac_e = (
        0.0 if lo_e == hi_e else (x - x_evals[lo_e]) / (x_evals[hi_e] - x_evals[lo_e])
    )
    for i in range(n):
        ctr_re = currents_evals[i, lo_e].real + frac_e * (
            currents_evals[i, hi_e].real - currents_evals[i, lo_e].real
        )
        ctr_im = currents_evals[i, lo_e].imag + frac_e * (
            currents_evals[i, hi_e].imag - currents_evals[i, lo_e].imag
        )
        ctr = (ctr_re + 1j * ctr_im) * en[i]
        ar[i] = currents[i] * ep[i]
        ar[i + n] = np.conj(ar[i])
        ar[i + 2 * n] = ctr
        ar[i + 3 * n] = np.conj(ctr)

    for idx, i1, i2 in relations_3wm:
        out[idx] += coeffs_3wm[idx] * ar[i1] * ar[i2]
    for idx, i1, i2, i3 in relations_4wm:
        out[idx] += coeffs_4wm[idx] * ar[i1] * ar[i2] * ar[i3]

    for i in range(n):
        raw = (alphas[i] + 1j * kappas[i]) * out[i] / ep[i]
        out[i] = raw + alphas[i] * currents[i]
        out[n + i] = raw

    return out


# Segmented RHS  (position-dependent parameters, exact cumulative phase)


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
        nb_complex2d,
        #   nb_complex2d,
    ),
    cache=True,
)
def cme_rhs_segmented(
    x,
    currents,
    alphas_table,
    kappas_table,
    relations_3wm,
    relations_4wm,
    coeffs_3wm_table,
    coeffs_4wm_table,
    segment_midpoints,
    x_evals,
    phi_table,
    currents_evals,
    #  Zb_table,
):
    """
    Unified CME RHS for segmented-device forward and backward passes.

    Phase carriers (forward and backward) are derived internally from
    ``phi_table`` rather than passed in directly.

    Parameters
    ----------
    alphas_table, kappas_table : ndarray, shape (n_modes, n_segments)
        Piecewise-linear attenuation and wavenumber, interpolated at ``x``
        via ``segment_midpoints``.
    coeffs_3wm_table, coeffs_4wm_table : ndarray, shape (n_relations, n_segments)
        Position-dependent nonlinear coupling coefficients.
    segment_midpoints : ndarray, shape (n_segments,)
        Sorted grid for the parameter tables.
    x_evals : ndarray, shape (n_x,)
        Sorted output grid on which ``phi_table`` and ``currents_evals`` are
        defined.
    phi_table : ndarray, shape (n_modes, n_x)
        Forward cumulative phase per mode, precomputed on ``x_evals``.
    currents_evals : ndarray, shape (n_modes, n_x)
        Reversed slow envelope of the counter-propagating wave. Zero on the
        first forward pass, so the counter term vanishes.
    Zb_table : ndarray, shape (n_modes, n_segments)
        Local Bloch impedance along the device, used for the distributed
        reflection gradient that couples forward and backward envelopes
        (WKB correction for slowly-varying impedance). Pass the reversed
        copy for the backward solve so the gradient sign flips correctly.

    Returns
    -------
    ndarray
        Array of length ``2 * n_modes``: the first half is ``dB_m/dx``
        (full, with the attenuation term), the second half is the raw
        nonlinear part only, used for the reflection-coefficient update.
    """
    n = len(currents)
    out = np.zeros(2 * n, nb.complex128)

    # ---- locate x in both sorted grids (one binary search each) --------- #
    lo_s, hi_s = _sorted_interp_bracket(segment_midpoints, x)
    lo_x, hi_x = _sorted_interp_bracket(x_evals, x)

    frac_s = (
        0.0
        if lo_s == hi_s
        else (x - segment_midpoints[lo_s])
        / (segment_midpoints[hi_s] - segment_midpoints[lo_s])
    )
    frac_x = (
        0.0 if lo_x == hi_x else (x - x_evals[lo_x]) / (x_evals[hi_x] - x_evals[lo_x])
    )

    # ---- local alphas, kappas, phase carriers ---------------------------- #
    # ep[m]  = exp(i * Phi_m(x))               forward carrier
    # en[m]  = exp(i * (Phi_m(x_end) - Phi_m(x)))  backward carrier
    #        = exp(i * phi_total[m]) / ep[m]
    # phi_total[m] = phi_table[m, -1]  (last column, computed once per call)
    alphas = np.empty(n, dtype=nb.float64)
    kappas = np.empty(n, dtype=nb.float64)
    ep = np.empty(n, dtype=nb.complex128)
    en = np.empty(n, dtype=nb.complex128)
    n_x_tab = phi_table.shape[1]
    for m in range(n):
        alphas[m] = alphas_table[m, lo_s] + frac_s * (
            alphas_table[m, hi_s] - alphas_table[m, lo_s]
        )
        kappas[m] = kappas_table[m, lo_s] + frac_s * (
            kappas_table[m, hi_s] - kappas_table[m, lo_s]
        )
        phi_x = phi_table[m, lo_x] + frac_x * (phi_table[m, hi_x] - phi_table[m, lo_x])
        phi_total = phi_table[m, n_x_tab - 1]  # Phi_m(x_end)
        ep[m] = np.exp(1j * phi_x)
        en[m] = np.exp(1j * (phi_total - phi_x))  # no extra allocation

    # ---- local nonlinear coefficients ------------------------------------ #
    n_rel3 = coeffs_3wm_table.shape[0]
    coeffs_3wm = np.empty(n_rel3, dtype=nb.complex128)
    for r in range(n_rel3):
        coeffs_3wm[r] = coeffs_3wm_table[r, lo_s] + frac_s * (
            coeffs_3wm_table[r, hi_s] - coeffs_3wm_table[r, lo_s]
        )

    n_rel4 = coeffs_4wm_table.shape[0]
    coeffs_4wm = np.empty(n_rel4, dtype=nb.complex128)
    for r in range(n_rel4):
        coeffs_4wm[r] = coeffs_4wm_table[r, lo_s] + frac_s * (
            coeffs_4wm_table[r, hi_s] - coeffs_4wm_table[r, lo_s]
        )

    # ---- assemble field amplitudes --------------------------------------- #
    # ar layout (same as uniform RHS):
    #   [0 :  n]  A_m(x)  = B_m * ep[m]         (co-propagating, full phase)
    #   [n : 2n]  conj(A_m)
    #   [2n: 3n]  C_m(x)  = D_m * en[m]         (counter-propagating, full phase)
    #   [3n: 4n]  conj(C_m)
    ar = np.empty(4 * n, dtype=nb.complex128)
    for i in range(n):
        ctr_re = currents_evals[i, lo_x].real + frac_x * (
            currents_evals[i, hi_x].real - currents_evals[i, lo_x].real
        )
        ctr_im = currents_evals[i, lo_x].imag + frac_x * (
            currents_evals[i, hi_x].imag - currents_evals[i, lo_x].imag
        )
        ctr = (ctr_re + 1j * ctr_im) * en[i]
        ar[i] = currents[i] * ep[i]
        ar[i + n] = np.conj(ar[i])
        ar[i + 2 * n] = ctr
        ar[i + 3 * n] = np.conj(ctr)

    # ---- nonlinear coupling ---------------------------------------------- #
    for idx, i1, i2 in relations_3wm:
        out[idx] += coeffs_3wm[idx] * ar[i1] * ar[i2]
    for idx, i1, i2, i3 in relations_4wm:
        out[idx] += coeffs_4wm[idx] * ar[i1] * ar[i2] * ar[i3]

    # ---- remove fast carrier; add loss term ------------------------------ #
    for i in range(n):
        raw = (alphas[i] + 1j * kappas[i]) * out[i] / ep[i]
        out[i] = raw + alphas[i] * currents[i]
        out[n + i] = raw

    # ---- distributed reflection from slowly-varying Zb(x) --------------- #
    # WKB correction: r(x) = (1/2) d ln Zb / dx ≈ (Zb[hi] - Zb[lo]) / (seg_len * Zb(x))
    # In current normalisation the equations are:
    #   dI⁺/dx += -r(x) * I⁺  +  r(x) * I⁻ * (en / ep)    [fwd slow envelope]
    #   dI⁻/dx += +r(x) * I⁺ * (ep / en)  -  r(x) * I⁻    [bwd slow envelope]
    # if lo_s != hi_s:
    #     seg_len = segment_midpoints[hi_s] - segment_midpoints[lo_s]
    #     for m in range(n):
    #         Zb_lo = Zb_table[m, lo_s]
    #         Zb_hi = Zb_table[m, hi_s]
    #         Zb_x  = Zb_lo + frac_s * (Zb_hi - Zb_lo)
    #         r_m   = 0.5 * (Zb_lo - Zb_hi) / (seg_len * Zb_x)
    #         refl_correction = r_m * (ar[m + 2 * n] / ep[m] - currents[m])
    #         out[m]     += refl_correction
    #         #out[m + n] += refl_correction

    return out


@nb.njit(inline="always")
def _boundary_reflection_update(
    raw_launch,
    B_launch,
    raw_inc,
    B_inc,
    gamma,
    Zb,
    Zout,
    Ze,
    apcz,
    refl_old,
    trans_old,
    update_rate,
):
    """
    Self-consistent boundary reflection/transmission update, one boundary, one mode.
    Takes into account the field derivative to modify the reflection coefficient and
    applies a finite update rate to avoid oscillations when reflections are strong.

    The field-derivative correction to Zb is a perturbative estimate,
    valid only while |raw_launch / (gamma * B_launch)| stays small. Close to a
    stopband edge with nonlinear interaction active, B_launch can pass through a
    near-node of the standing-wave pattern while raw_launch stays finite, which
    pushes this ratio to O(1) or larger and can drive refl_new outside the
    physical |Gamma| <= 1 regime. Two safeguards are applied:
      1. the correction ratio is saturated at BOUNDARY_CORR_MAX (keeps the
         approximation perturbative by construction, smoothly, instead of
         only guarding against the exact-zero division), and
      2. refl_new is projected onto the physical |Gamma| <= 1 disk before
         being mixed into the running estimate, since Zout is passive.
    """
    gB_launch = gamma * B_launch
    gB_inc = gamma * B_inc

    if abs(gB_launch) < 1e-20 or abs(gB_inc) < 1e-20:
        Ztilde_launch = Zb
        refl_new = -(Zout - Zb) / (Zout + Zb)
    else:
        corr_launch = raw_launch / gB_launch
        corr_launch_mag = abs(corr_launch)
        if corr_launch_mag > BOUNDARY_CORR_MAX:
            corr_launch = corr_launch * (BOUNDARY_CORR_MAX / corr_launch_mag)
        Ztilde_launch = Zb * (1.0 + corr_launch)

        corr_inc = raw_inc / gB_inc
        corr_inc_mag = abs(corr_inc)
        if corr_inc_mag > BOUNDARY_CORR_MAX:
            corr_inc = corr_inc * (BOUNDARY_CORR_MAX / corr_inc_mag)
        Ztilde_inc = Zb * (1.0 + corr_inc)

        refl_new = -(Zout - Ztilde_launch) / (Zout + Ztilde_inc)
    refl_mag = abs(refl_new)
    if refl_mag > 1.0:
        refl_new = refl_new / refl_mag

    trans_new = 2.0 * Ze / (apcz * (Ztilde_launch + Zout))

    refl = refl_old + update_rate * (refl_new - refl_old)
    trans = trans_old + update_rate * (trans_new - trans_old)
    return refl, trans


@nb.njit(inline="always")
def _init_boundary_state(
    Zb_0,
    Zb_end,
    Zout_in,
    Zout_out,
    Ze,
    apcz_in,
    apcz_out,
    y0_fwd,
    y0_bwd,
    n_modes,
    trans_0_in,
    trans_end_in,
    refl_0_out,
    refl_end_out,
    I_fwd,
    I_bwd,
):
    """Linear-limit (Ztilde=Zb) boundary transmissions/reflections and launch currents."""
    for j in range(n_modes):
        trans_0_in[j] = 2.0 * Ze[j] / (apcz_in[j] * (Zb_0[j] + Zout_in[j]))
        trans_end_in[j] = 2.0 * Ze[j] / (apcz_out[j] * (Zb_end[j] + Zout_out[j]))
        refl_end_out[j] = (Zb_end[j] - Zout_out[j]) / (Zb_end[j] + Zout_out[j])
        refl_0_out[j] = (Zb_0[j] - Zout_in[j]) / (Zb_0[j] + Zout_in[j])
        I_fwd[j] = y0_fwd[j] * trans_0_in[j]
        I_bwd[j] = y0_bwd[j] * trans_end_in[j]


@nb.njit(inline="always")
def _compute_backward_launch(
    A_fwd_i, y0_bwd, trans_end_in, refl_end_out, phase_end, I_bwd, n_modes
):
    """Backward-solve initial condition from the reflected forward field at x_end."""
    for j in range(n_modes):
        A_fwd_end_j = A_fwd_i[j, -1] * phase_end[j]
        I_bwd[j] = y0_bwd[j] * trans_end_in[j] + refl_end_out[j] * A_fwd_end_j


@nb.njit(inline="always")
def _update_output_boundary(
    A_fwd_i,
    A_bwd_i,
    gam,
    Zb,
    Zout_out,
    Ze,
    apcz_out,
    refl_end_out,
    trans_end_in,
    update_rate,
    n_modes,
):
    """Self-consistent reflection/transmission update at the output (x=L) boundary."""
    for j in range(n_modes):
        raw_bwd = A_bwd_i[n_modes + j, 0]
        B_bwd0 = A_bwd_i[j, 0]
        raw_fwd = A_fwd_i[n_modes + j, -1]
        B_fwd_end = A_fwd_i[j, -1]
        refl_end_out[j], trans_end_in[j] = _boundary_reflection_update(
            raw_bwd,
            B_bwd0,
            raw_fwd,
            B_fwd_end,
            gam[j],
            Zb[j],
            Zout_out[j],
            Ze[j],
            apcz_out[j],
            refl_end_out[j],
            trans_end_in[j],
            update_rate,
        )


@nb.njit(inline="always")
def _update_input_boundary_and_launch(
    A_fwd_i,
    A_bwd_i,
    gam,
    Zb,
    Zout_in,
    Ze,
    apcz_in,
    refl_0_out,
    trans_0_in,
    update_rate,
    y0_fwd,
    phase_end,
    I_fwd,
    n_modes,
    do_update,
):
    """Self-consistent reflection/transmission update at the input (x=0) boundary
    and the resulting forward-solve initial condition for the next pass.
    ``do_update`` gates the reflection update itself (only run for k > 1,
    matching the linear-limit fallback used on the first two passes) while
    the launch current is always recomputed from the current refl/trans state.
    """
    for j in range(n_modes):
        if do_update:
            raw_fwd = A_fwd_i[n_modes + j, 0]
            B_fwd0 = A_fwd_i[j, 0]
            raw_bwd = A_bwd_i[n_modes + j, -1]
            B_bwd_end = A_bwd_i[j, -1]
            refl_0_out[j], trans_0_in[j] = _boundary_reflection_update(
                raw_fwd,
                B_fwd0,
                raw_bwd,
                B_bwd_end,
                gam[j],
                Zb[j],
                Zout_in[j],
                Ze[j],
                apcz_in[j],
                refl_0_out[j],
                trans_0_in[j],
                update_rate,
            )
        A_bwd_end_j = A_bwd_i[j, -1] * phase_end[j]
        I_fwd[j] = y0_fwd[j] * trans_0_in[j] + refl_0_out[j] * A_bwd_end_j


@nb.njit(inline="always")
def _convergence_step(
    A_fwd_i, A_bwd_i, prev_fwd, prev_bwd, n_modes, convergence_threshold, stable_count
):
    """Relative-change convergence test; returns updated (stable_count, converged)."""
    nf = 0.0
    nb_ = 0.0
    for j in range(n_modes):
        v = np.abs(A_fwd_i[j, -1] - prev_fwd[j]) / (np.abs(A_fwd_i[j, -1]) + 1e-30)
        w = np.abs(A_bwd_i[j, -1] - prev_bwd[j]) / (np.abs(A_bwd_i[j, -1]) + 1e-30)
        if v > nf:
            nf = v
        if w > nb_:
            nb_ = w
    if nf < convergence_threshold and nb_ < convergence_threshold:
        stable_count += 1
    else:
        stable_count = 0
    converged = stable_count >= 3
    return stable_count, converged


@nb.njit(inline="always")
def _store_pass(
    I_out,
    A_fwd_i,
    A_bwd_i,
    refl_end_out,
    refl_0_out,
    tof_out,
    tof_in,
    idx_fwd,
    idx_bwd,
    n_modes,
):
    """Write one forward/backward pass into I_out, overwriting the last x-point
    with the physical outcoupled current: forward-out (S21) at idx_fwd,
    backward-out (S11) at idx_bwd.
    """
    I_out[idx_fwd] = A_fwd_i[:n_modes, :]
    I_out[idx_bwd] = A_bwd_i[:n_modes, :]
    for j in range(n_modes):
        I_out[idx_fwd, j, -1] = A_fwd_i[j, -1] * (1.0 + refl_end_out[j]) * tof_out[j]
        I_out[idx_bwd, j, -1] = A_bwd_i[j, -1] * (1.0 + refl_0_out[j]) * tof_in[j]


@nb.njit(parallel=True)
def _general_cmes_solve_fb_cutoff_uniform(
    x_array,
    y0_array_fwd,
    y0_array_bwd,
    gammas_array,
    Zb_table,
    relations_3wm,
    relations_4wm,
    coeffs_3wm,
    coeffs_4wm,
    passes,
    kappa_cutoff,
    update_rate,
    convergence_threshold,
    save_all_passes,
    Ze_table,
    Zout_in_table,
    Zout_out_table,
    apcz_in_table,
    apcz_out_table,
    tof_in_table,
    tof_out_table,
):
    """
    Solve full general CMEs (forward + backward) for all frequency points.

    Two-port boundary model (same as the segmented kernel; see
    ``_general_cmes_solve_fb_cutoff_segmented`` and ``_boundary_reflection_update``
    for the full derivation).  A uniform device is the degenerate case of the
    segmented boundary formalism with a single, constant Zb/gamma per mode
    (no per-segment tables), so the same shared helper is reused directly.
    """
    n_freq = gammas_array.shape[0]
    n_modes = gammas_array.shape[1]
    n_out = 2 * n_modes
    n_x = len(x_array)
    x_end = x_array[-1]
    x_span = (x_array[0], x_array[-1])

    if save_all_passes:
        I_out = np.empty((2 * n_freq * passes, n_modes, n_x), dtype=np.complex128)
    else:
        I_out = np.empty((2 * n_freq, n_modes, n_x), dtype=np.complex128)

    A_fwd = np.zeros((n_freq, n_out, n_x), dtype=np.complex128)
    A_bwd = np.zeros((n_freq, n_out, n_x), dtype=np.complex128)
    A_counter = np.zeros((n_freq, n_modes, n_x), dtype=np.complex128)

    for i in nb.prange(n_freq):
        I_fwd = np.zeros(n_modes, dtype=np.complex128)
        I_bwd = np.zeros(n_modes, dtype=np.complex128)
        refl_end_out = np.zeros(n_modes, dtype=np.complex128)
        refl_0_out = np.zeros(n_modes, dtype=np.complex128)

        converged = False
        stable_count = 0
        prev_fwd = np.zeros(n_modes, dtype=np.complex128)
        prev_bwd = np.zeros(n_modes, dtype=np.complex128)

        alphas_i = np.ascontiguousarray(gammas_array[i].real)
        kappas_i = np.ascontiguousarray(gammas_array[i].imag)
        Zb_i = Zb_table[i]
        gam_i = gammas_array[i]

        # ---- boundary two-port tables for this frequency ---------------- #
        Ze_i = Ze_table[i]
        Zout_in_i = Zout_in_table[i]
        Zout_out_i = Zout_out_table[i]
        apcz_in = apcz_in_table[i]
        apcz_out = apcz_out_table[i]
        tof_in = tof_in_table[i]
        tof_out = tof_out_table[i]

        x_array_i = np.ascontiguousarray(x_array)

        # Full-phase-at-x_end carrier, shared by both boundary launches. A
        # uniform device is the degenerate single-segment case of the
        # segmented formalism, so "the endpoint" is just (Zb_i, gam_i)
        # themselves -- same shared helpers as the segmented kernel below.
        phase_end = np.exp(1j * kappas_i * x_end)

        trans_0_in = np.empty(n_modes, dtype=nb.complex128)
        trans_end_in = np.empty(n_modes, dtype=nb.complex128)
        _init_boundary_state(
            Zb_i,
            Zb_i,
            Zout_in_i,
            Zout_out_i,
            Ze_i,
            apcz_in,
            apcz_out,
            y0_array_fwd[i],
            y0_array_bwd[i],
            n_modes,
            trans_0_in,
            trans_end_in,
            refl_0_out,
            refl_end_out,
            I_fwd,
            I_bwd,
        )

        r3, r4 = _filter_wm_terms(kappas_i, relations_3wm, relations_4wm, kappa_cutoff)

        for k in range(passes):
            # ---- forward solve ------------------------------------------ #
            A_fwd[i] = nbsolve_ivp(
                cme_rhs,
                x_span,
                I_fwd,
                args=(
                    alphas_i,
                    kappas_i,
                    r3,
                    r4,
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
                capture_extra=True,
            ).y

            # ---- reflection at far end → backward IC -------------------- #
            _compute_backward_launch(
                A_fwd[i],
                y0_array_bwd[i],
                trans_end_in,
                refl_end_out,
                phase_end,
                I_bwd,
                n_modes,
            )

            # Reverse A_fwd into counter buffer for the backward solve.
            for j in range(n_modes):
                for ix in range(n_x):
                    A_counter[i, j, ix] = A_fwd[i, j, n_x - 1 - ix]

            # ---- backward solve ----------------------------------------- #
            A_bwd[i] = nbsolve_ivp(
                cme_rhs,
                x_span,
                I_bwd,
                args=(
                    alphas_i,
                    kappas_i,
                    r3,
                    r4,
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
                capture_extra=True,
            ).y

            # ---- self-consistent reflection update (output end → bwd IC) #
            # launch = backward wave (into TWPA), inc = forward wave (arriving).
            if k > 1:
                _update_output_boundary(
                    A_fwd[i],
                    A_bwd[i],
                    gam_i,
                    Zb_i,
                    Zout_out_i,
                    Ze_i,
                    apcz_out,
                    refl_end_out,
                    trans_end_in,
                    update_rate,
                    n_modes,
                )

            # Reverse A_bwd into counter buffer for the next forward solve.
            for j in range(n_modes):
                for ix in range(n_x):
                    A_counter[i, j, ix] = A_bwd[i, j, n_x - 1 - ix]

            # ---- self-consistent reflection update (input end → fwd IC) -#
            # launch = forward wave (into TWPA), inc = backward wave (arriving).
            _update_input_boundary_and_launch(
                A_fwd[i],
                A_bwd[i],
                gam_i,
                Zb_i,
                Zout_in_i,
                Ze_i,
                apcz_in,
                refl_0_out,
                trans_0_in,
                update_rate,
                y0_array_fwd[i],
                phase_end,
                I_fwd,
                n_modes,
                k > 1,
            )

            # ---- convergence check -------------------------------------- #
            if k > 1:
                stable_count, converged = _convergence_step(
                    A_fwd[i],
                    A_bwd[i],
                    prev_fwd,
                    prev_bwd,
                    n_modes,
                    convergence_threshold,
                    stable_count,
                )

            # ---- store output ------------------------------------------- #
            # Last x-point overwritten with the physical outcoupled current:
            #   forward-out (S21, output end): A_fwd(L) * (1+rho_out) * tof_out
            #   backward-out (S11, input end): A_bwd(0) * (1+rho_in)  * tof_in
            id_add_fwd = 0
            id_add_bwd = n_freq
            if save_all_passes:
                id_add_fwd = k * n_freq
                id_add_bwd = k * n_freq + n_freq * passes
            _store_pass(
                I_out,
                A_fwd[i],
                A_bwd[i],
                refl_end_out,
                refl_0_out,
                tof_out,
                tof_in,
                i + id_add_fwd,
                i + id_add_bwd,
                n_modes,
            )

            for j in range(n_modes):
                prev_fwd[j] = A_fwd[i, j, -1]
                prev_bwd[j] = A_bwd[i, j, -1]

            # ---- early exit on convergence ------------------------------ #
            if converged:
                if save_all_passes:
                    for kk in range(k + 1, passes):
                        _store_pass(
                            I_out,
                            A_fwd[i],
                            A_bwd[i],
                            refl_end_out,
                            refl_0_out,
                            tof_out,
                            tof_in,
                            i + kk * n_freq,
                            i + kk * n_freq + n_freq * passes,
                            n_modes,
                        )
                break

    return I_out


# Parallel outer loop — segmented device


@nb.njit(parallel=True)
def _general_cmes_solve_fb_cutoff_segmented(
    x_array,
    y0_array_fwd,
    y0_array_bwd,
    gammas_array,
    Zb_table,
    relations_3wm,
    relations_4wm,
    coeffs_3wm_table,
    coeffs_4wm_table,
    segment_midpoints,
    passes,
    kappa_cutoff,
    update_rate,
    convergence_threshold,
    save_all_passes,
    Ze_table,
    Zout_in_table,
    Zout_out_table,
    apcz_in_table,
    apcz_out_table,
    tof_in_table,
    tof_out_table,
):
    """
    Forward/backward CME solver for segmented (position-dependent) devices.

    Two-port boundary model
    -----------------------
    Each boundary carries a generic reciprocal/non-reciprocal two-port
    (port 1 -> external cable Ze, port 2 -> TWPA) described by its ABCD matrix.
    All four boundary coefficients follow from voltage/current continuity on
    the *total* field with each traveling wave carrying its own generalized
    current impedance

        Ztilde_wave = Zb * (1 + raw_wave / (gamma * B_wave)),

    where ``raw`` is the pure nonlinear derivative (rows [n:2n] of the RHS) and
    ``B`` the slow envelope.  With  W = (A + C*Ze)*Ztilde_launch + (B + D*Ze),
    Zin = (A*Ztilde_launch + B)/(C*Ztilde_launch + D)  and
    Zout = (D*Ze + B)/(C*Ze + A):

      t_in      = 2*Ze / W                      (cable  -> launched TWPA wave)
      Gamma_in  = (Ze - Zin)/(Ze + Zin)         (cable  -> reflected cable, prompt)
      rho       = (Ztilde_launch - Zout)
                  /(Ztilde_inc   + Zout)         (arriving TWPA wave -> re-launched)
      t_out     = det * (Ztilde_launch + Ztilde_inc)/W
                = (det/(A+C*Ze)) * (1 + rho)     (arriving TWPA wave -> cable)

    with det = A*D - B*C.  "launch" is the wave sent *into* the TWPA at that
    boundary (forward at x=0, backward at x=L); "inc" is the internal wave
    arriving *from* the TWPA (backward at x=0, forward at x=L).  Identity ABCD
    (A=D=1, B=C=0, Ze=Zout) plus pump-off (Ztilde->Zb) recover the bare
    single-impedance boundary.

    The field-independent ABCD scalars ``apcz = A + C*Ze`` and
    ``tof = det/(A + C*Ze)`` (the current-transfer factor multiplying 1+rho)
    are precomputed once in plain NumPy by the dispatcher (they don't depend
    on the self-consistent iteration), rather than recomputed per-thread here.

    Parameters (boundary tables; all sliced per frequency i)
    --------------------------------------------------------
    Ze_table       : (n_freq, n_modes) complex   external cable impedance.
    Zout_in_table  : (n_freq, n_modes) complex   (D*Ze+B)/(C*Ze+A) at x=0.
    Zout_out_table : (n_freq, n_modes) complex   (D*Ze+B)/(C*Ze+A) at x=L.
    apcz_in_table  : (n_freq, n_modes) complex   A+C*Ze of input connector.
    apcz_out_table : (n_freq, n_modes) complex   A+C*Ze of output connector.
    tof_in_table   : (n_freq, n_modes) complex   det/(A+C*Ze) of input connector.
    tof_out_table  : (n_freq, n_modes) complex   det/(A+C*Ze) of output connector.

    Parameters
    ----------
    gammas_array : (n_freq, n_modes, n_segments)
        Complex propagation constants along the device.
    Zb_table : (n_freq, n_modes, n_segments)
        Bloch impedance table.  Boundary reflections and transmission
        coefficients are evaluated at the endpoint columns:
          - input  end: Zb_table[i, j, 0]   (column 0)
          - output end: Zb_table[i, j, -1]  (last column)
    coeffs_3wm_table, coeffs_4wm_table : (n_relations, n_segments)
        Position-dependent coupling coefficients.
    segment_midpoints : (n_segments,)
        Sorted positions at which the parameter tables are defined.

    Phase convention
    ----------------
    phi_table[m, k] = integral_0^{x[k]} kappa_m(x') dx'   (forward phase)

    Both RHS calls receive this same phi_table.  Inside cme_rhs_segmented:
      - forward  carrier: exp(i * phi_table[m, x])
      - backward carrier: exp(i * (phi_table[m, x_end] - phi_table[m, x]))
        derived from phi_total[m] = phi_table[m, -1], so no separate table
        is needed.

    The counter-propagating envelope stored in A_counter is the slow
    envelope reversed in x; cme_rhs_segmented restores the full-phase field
    via the backward carrier above.
    """
    n_freq = gammas_array.shape[0]
    n_modes = gammas_array.shape[1]
    n_out = 2 * n_modes
    n_x = len(x_array)
    x_span = (x_array[0], x_array[-1])

    if save_all_passes:
        I_out = np.empty((2 * n_freq * passes, n_modes, n_x), dtype=np.complex128)
    else:
        I_out = np.empty((2 * n_freq, n_modes, n_x), dtype=np.complex128)

    A_fwd = np.zeros((n_freq, n_out, n_x), dtype=np.complex128)
    A_bwd = np.zeros((n_freq, n_out, n_x), dtype=np.complex128)
    A_counter = np.zeros((n_freq, n_modes, n_x), dtype=np.complex128)

    for i in nb.prange(n_freq):
        alphas_i = np.ascontiguousarray(gammas_array[i].real)  # (n_modes, n_seg)
        kappas_i = np.ascontiguousarray(gammas_array[i].imag)  # (n_modes, n_seg)

        # ---- boundary two-port tables for this frequency ---------------- #
        # apcz/tof are field-independent ABCD scalars, precomputed once in
        # plain NumPy by the dispatcher (they don't depend on the
        # self-consistent iteration) -- just slice per frequency here.
        Ze_i = Ze_table[i]  # (n_modes,) external cable impedance
        Zout_in_i = Zout_in_table[i]  # (n_modes,) network seen from TWPA @ x=0
        Zout_out_i = Zout_out_table[i]  # (n_modes,) network seen from TWPA @ x=L
        apcz_in = apcz_in_table[i]  # (n_modes,) A+C*Ze, input connector
        apcz_out = apcz_out_table[i]  # (n_modes,) A+C*Ze, output connector
        tof_in = tof_in_table[i]  # (n_modes,) det/(A+C*Ze), input connector
        tof_out = tof_out_table[i]  # (n_modes,) det/(A+C*Ze), output connector

        # Endpoint Zb and gamma used for boundary conditions.
        # Input end:  column 0;  output end: column -1.
        n_seg = alphas_i.shape[1]
        Zb_0 = np.ascontiguousarray(Zb_table[i, :, 0])
        Zb_end = np.ascontiguousarray(Zb_table[i, :, n_seg - 1])
        gam_0 = np.ascontiguousarray(gammas_array[i, :, 0])
        gam_end = np.ascontiguousarray(gammas_array[i, :, n_seg - 1])

        # Precompute cumulative forward phase on the output grid.
        # phi_table[m, k] = Phi_m(x[k]) = integral_0^{x[k]} kappa_m(x') dx'.
        # kappa_m(x) is piecewise-linear between segment_midpoints, so the
        # trapezoid rule on the linearly-interpolated values is exact.
        phi_table = np.zeros((n_modes, n_x), dtype=nb.float64)
        phi_table_bwd = np.zeros((n_modes, n_x), dtype=nb.float64)
        for kx in range(1, n_x):
            lo_l, hi_l = _sorted_interp_bracket(segment_midpoints, x_array[kx - 1])
            fl = (
                0.0
                if lo_l == hi_l
                else (x_array[kx - 1] - segment_midpoints[lo_l])
                / (segment_midpoints[hi_l] - segment_midpoints[lo_l])
            )
            lo_r, hi_r = _sorted_interp_bracket(segment_midpoints, x_array[kx])
            fr = (
                0.0
                if lo_r == hi_r
                else (x_array[kx] - segment_midpoints[lo_r])
                / (segment_midpoints[hi_r] - segment_midpoints[lo_r])
            )
            dx = x_array[kx] - x_array[kx - 1]
            for m in range(n_modes):
                kl = kappas_i[m, lo_l] + fl * (kappas_i[m, hi_l] - kappas_i[m, lo_l])
                kr = kappas_i[m, lo_r] + fr * (kappas_i[m, hi_r] - kappas_i[m, lo_r])
                phi_table[m, kx] = phi_table[m, kx - 1] + 0.5 * (kl + kr) * dx
        # phi_table[m, -1] = Phi_m(x_end): used by cme_rhs_segmented for en.

        # phi_total[m] = Phi_m(x_end), used for boundary IC phase restoration
        # (the "phase_end" carrier shared with the uniform kernel's helpers).
        # phi_table_bwd[m, k] = cumulative phase for the backward solve.
        # The backward solve uses a mirrored coordinate  s = x_end - x  so that
        # s runs from 0 (physical x_end) to x_end (physical x=0) in the same
        # ascending direction the ODE integrator expects.
        # phi_table_bwd[m, k] = Phi_m(x_end) - Phi_m(x_array[n_x-1-k])
        # = total accumulated phase minus the phase at the mirrored position.
        phi_total = np.empty(n_modes, dtype=nb.float64)
        for m in range(n_modes):
            phi_total[m] = phi_table[m, n_x - 1]
            for kx in range(n_x):
                phi_table_bwd[m, kx] = phi_total[m] - phi_table[m, n_x - 1 - kx]
        phase_end = np.exp(1j * phi_total)

        # Reversed coordinate grid for the backward solve.
        # x_array_bwd[k] = x_end - x_array[n_x-1-k]  (ascending 0→x_end).
        # Parameter tables are reversed along the segment axis once here,
        # outside the pass loop, so the RHS inner loop stays branchless.
        x_end_val = x_array[n_x - 1]
        x_array_bwd = np.empty(n_x, dtype=nb.float64)
        for kx in range(n_x):
            x_array_bwd[kx] = x_end_val - x_array[n_x - 1 - kx]

        segment_midpoints_bwd = np.empty(n_seg, dtype=nb.float64)
        alphas_i_bwd = np.empty((n_modes, n_seg), dtype=nb.float64)
        kappas_i_bwd = np.empty((n_modes, n_seg), dtype=nb.float64)
        n_rel3 = coeffs_3wm_table.shape[0]
        n_rel4 = coeffs_4wm_table.shape[0]
        coeffs_3wm_bwd = np.empty((n_rel3, n_seg), dtype=nb.complex128)
        coeffs_4wm_bwd = np.empty((n_rel4, n_seg), dtype=nb.complex128)
        for s in range(n_seg):
            s_rev = n_seg - 1 - s
            segment_midpoints_bwd[s] = x_end_val - segment_midpoints[s_rev]
            for m in range(n_modes):
                alphas_i_bwd[m, s] = alphas_i[m, s_rev]
                kappas_i_bwd[m, s] = kappas_i[m, s_rev]
            for r in range(n_rel3):
                coeffs_3wm_bwd[r, s] = coeffs_3wm_table[r, s_rev]
            for r in range(n_rel4):
                coeffs_4wm_bwd[r, s] = coeffs_4wm_table[r, s_rev]

        # Use midpoint kappas for the wave-mixing filter (conservative: keeps
        # terms that are phase-matched anywhere in the device).
        mid_seg = n_seg // 2
        kappas_mid = np.empty(n_modes, dtype=nb.float64)
        for m in range(n_modes):
            kappas_mid[m] = np.mean(kappas_i[m])
        r3, r4 = _filter_wm_terms(
            kappas_mid, relations_3wm, relations_4wm, kappa_cutoff
        )

        I_fwd = np.zeros(n_modes, dtype=np.complex128)
        I_bwd = np.zeros(n_modes, dtype=np.complex128)
        refl_end_out = np.zeros(n_modes, dtype=np.complex128)
        refl_0_out = np.zeros(n_modes, dtype=np.complex128)

        converged = False
        stable_count = 0
        prev_fwd = np.zeros(n_modes, dtype=np.complex128)
        prev_bwd = np.zeros(n_modes, dtype=np.complex128)

        trans_0_in = np.empty(n_modes, dtype=nb.complex128)
        trans_end_in = np.empty(n_modes, dtype=nb.complex128)
        _init_boundary_state(
            Zb_0,
            Zb_end,
            Zout_in_i,
            Zout_out_i,
            Ze_i,
            apcz_in,
            apcz_out,
            y0_array_fwd[i],
            y0_array_bwd[i],
            n_modes,
            trans_0_in,
            trans_end_in,
            refl_0_out,
            refl_end_out,
            I_fwd,
            I_bwd,
        )

        x_array_i = np.ascontiguousarray(x_array)

        atol_fwd = np.full(n_modes, SOLVER_ATOL)
        atol_bwd = np.full(n_modes, SOLVER_ATOL)

        for k in range(passes):
            # ---- forward solve ------------------------------------------ #
            A_fwd[i] = nbsolve_ivp(
                cme_rhs_segmented,
                x_span,
                I_fwd,
                args=(
                    alphas_i,
                    kappas_i,
                    r3,
                    r4,
                    coeffs_3wm_table,
                    coeffs_4wm_table,
                    segment_midpoints,
                    x_array_i,
                    phi_table,
                    A_counter[i],
                ),
                atols=atol_fwd,
                rtol=SOLVER_RTOL,
                max_num_steps=SOLVER_MAX_STEPS,
                t_eval=x_array_i,
                rk_method=SOLVER_RK_METHOD,
                first_step=SOLVER_FIRST_STEP,
                capture_extra=True,
            ).y

            # ---- reflection at output end → backward IC ----------------- #
            _compute_backward_launch(
                A_fwd[i],
                y0_array_bwd[i],
                trans_end_in,
                refl_end_out,
                phase_end,
                I_bwd,
                n_modes,
            )

            # Reverse A_fwd slow envelope into counter buffer (fused loop).
            for j in range(n_modes):
                for ix in range(n_x):
                    A_counter[i, j, ix] = A_fwd[i, j, n_x - 1 - ix]

            # ---- backward solve ----------------------------------------- #
            # All parameter tables are pre-reversed (segment axis flipped) so
            # the RHS reads physical positions from the output end first as the
            # solver steps from s=0 to s=x_end.  No branching in the inner loop.
            A_bwd[i] = nbsolve_ivp(
                cme_rhs_segmented,
                x_span,
                I_bwd,
                args=(
                    alphas_i_bwd,
                    kappas_i_bwd,
                    r3,
                    r4,
                    coeffs_3wm_bwd,
                    coeffs_4wm_bwd,
                    segment_midpoints_bwd,
                    x_array_bwd,
                    phi_table_bwd,
                    A_counter[i],
                ),
                atols=atol_bwd,
                rtol=SOLVER_RTOL,
                max_num_steps=SOLVER_MAX_STEPS,
                t_eval=x_array_bwd,
                rk_method=SOLVER_RK_METHOD,
                first_step=SOLVER_FIRST_STEP,
                capture_extra=True,
            ).y

            # ---- self-consistent reflection update at output end -------- #
            # launch = backward wave (into TWPA), inc = forward wave (arriving).
            if k > 1:
                _update_output_boundary(
                    A_fwd[i],
                    A_bwd[i],
                    gam_end,
                    Zb_end,
                    Zout_out_i,
                    Ze_i,
                    apcz_out,
                    refl_end_out,
                    trans_end_in,
                    update_rate,
                    n_modes,
                )

            # Reverse A_bwd slow envelope into counter buffer (fused loop).
            for j in range(n_modes):
                for ix in range(n_x):
                    A_counter[i, j, ix] = A_bwd[i, j, n_x - 1 - ix]

            # ---- self-consistent reflection update at input end --------- #
            # launch = forward wave (into TWPA), inc = backward wave (arriving).
            _update_input_boundary_and_launch(
                A_fwd[i],
                A_bwd[i],
                gam_0,
                Zb_0,
                Zout_in_i,
                Ze_i,
                apcz_in,
                refl_0_out,
                trans_0_in,
                update_rate,
                y0_array_fwd[i],
                phase_end,
                I_fwd,
                n_modes,
                k > 1,
            )

            # ---- update per-mode atol vectors after the first pass ------- #
            if k == 0:
                atol_fwd = _compute_atol_vec(A_fwd[i], n_modes, SOLVER_ATOL)
                atol_bwd = _compute_atol_vec(A_bwd[i], n_modes, SOLVER_ATOL)

            # ---- convergence check -------------------------------------- #
            if k > 1:
                stable_count, converged = _convergence_step(
                    A_fwd[i],
                    A_bwd[i],
                    prev_fwd,
                    prev_bwd,
                    n_modes,
                    convergence_threshold,
                    stable_count,
                )

            # ---- store output ------------------------------------------- #
            # Last x-point overwritten with the physical outcoupled current:
            #   forward-out (S21, output end): A_fwd(L) * (1+rho_out) * tof_out
            #   backward-out (S11, input end): A_bwd(0) * (1+rho_in)  * tof_in
            id_add_fwd = 0
            id_add_bwd = n_freq
            if save_all_passes:
                id_add_fwd = k * n_freq
                id_add_bwd = k * n_freq + n_freq * passes

            _store_pass(
                I_out,
                A_fwd[i],
                A_bwd[i],
                refl_end_out,
                refl_0_out,
                tof_out,
                tof_in,
                i + id_add_fwd,
                i + id_add_bwd,
                n_modes,
            )

            for j in range(n_modes):
                prev_fwd[j] = A_fwd[i, j, -1]
                prev_bwd[j] = A_bwd[i, j, -1]

            # ---- early exit on convergence ------------------------------ #
            if converged:
                if save_all_passes:
                    for kk in range(k + 1, passes):
                        _store_pass(
                            I_out,
                            A_fwd[i],
                            A_bwd[i],
                            refl_end_out,
                            refl_0_out,
                            tof_out,
                            tof_in,
                            i + kk * n_freq,
                            i + kk * n_freq + n_freq * passes,
                            n_modes,
                        )
                break
    return I_out


# Public Python-level dispatcher


def general_cmes_solve_fb_cutoff(
    x_array: FloatArray,
    y0_array_fwd: ComplexArray,
    y0_array_bwd: ComplexArray,
    gammas_array: ComplexArray,
    Zb_table: ComplexArray,
    relations_3wm,
    relations_4wm,
    coeffs_3wm,
    coeffs_4wm,
    passes: int = 3,
    kappa_cutoff: float = 0.1,
    update_rate: float = 0.95,
    convergence_threshold: float = 0.05,
    save_all_passes: bool = False,
    Z0: float | ComplexArray = 50,
    segment_midpoints: FloatArray | None = None,
    Ze: float | ComplexArray | None = None,
    Zout_in: ComplexArray | None = None,
    Zout_out: ComplexArray | None = None,
    abcd_in: ComplexArray | None = None,
    abcd_out: ComplexArray | None = None,
) -> ComplexArray:
    """
    Solve the full forward/backward CME system for every signal frequency.

    Dispatches to the segmented or uniform Numba kernel based on
    ``gammas_array.ndim``:

    - Uniform: ``gammas_array`` and ``Zb_table`` shape (n_freq, n_modes);
      ``coeffs_*`` 1-D.
    - Segmented: ``gammas_array`` and ``Zb_table`` shape
      (n_freq, n_modes, n_seg); ``coeffs_*`` shape (n_rel, n_seg);
      ``segment_midpoints`` required.

    Two-port boundaries (both paths)

    Both kernels model each boundary as a generic two-port (port 1 ->
    external cable ``Ze``, port 2 -> TWPA), described by:

    - ``Ze``: scalar or (n_freq, n_modes) external cable impedance.
    - ``Zout_in``, ``Zout_out``: (n_freq, n_modes) two-port input impedance
      at x=0 and x=L.
    - ``abcd_in``, ``abcd_out``: (n_freq, n_modes, 4) ``[A, B, C, D]``
      connector parameters.

    These match the tables produced by ``TWPAnalysis._build_port_tables``.
    When left as None, identity connectors are used and
    ``Ze = Zout_in = Zout_out = Z0``, recovering the bare boundary.

    ``Z0`` (used for the uniform path and as the default external
    reference) may be a scalar float or a 2-D complex array of shape
    (n_freq, n_modes).
    """
    n_freq, n_modes = gammas_array.shape[0], gammas_array.shape[1]
    # Normalise Z0 to a (n_freq, n_modes) complex128 array.
    if np.ndim(Z0) == 0:
        Z0_table_arg = np.full((n_freq, n_modes), complex(Z0), dtype=np.complex128)
    else:
        Z0_table_arg = np.asarray(Z0, dtype=np.complex128)
        if Z0_table_arg.shape != (n_freq, n_modes):
            raise ValueError(
                f"Z0 array must have shape (n_freq, n_modes)=({n_freq}, {n_modes}), "
                f"got {Z0_table_arg.shape}."
            )

    def _as_nfm(val, default_table):
        """Broadcast a scalar/array to a (n_freq, n_modes) complex128 table."""
        if val is None:
            return default_table
        if np.ndim(val) == 0:
            return np.full((n_freq, n_modes), complex(val), dtype=np.complex128)
        arr = np.asarray(val, dtype=np.complex128)
        if arr.shape != (n_freq, n_modes):
            raise ValueError(
                f"Boundary table must have shape (n_freq, n_modes)="
                f"({n_freq}, {n_modes}), got {arr.shape}."
            )
        return arr

    # ---- assemble boundary two-port tables (shared by both kernels) ----- #
    # Defaults (no connector): Ze = Zout = Z0, identity ABCD.
    Ze_table = _as_nfm(Ze, Z0_table_arg)
    Zout_in_table = _as_nfm(Zout_in, Ze_table)
    Zout_out_table = _as_nfm(Zout_out, Ze_table)

    abcd_identity = np.zeros((n_freq, n_modes, 4), dtype=np.complex128)
    abcd_identity[:, :, 0] = 1.0  # A
    abcd_identity[:, :, 3] = 1.0  # D
    if abcd_in is None:
        abcd_in_table = abcd_identity
    else:
        abcd_in_table = np.asarray(abcd_in, dtype=np.complex128)
    if abcd_out is None:
        abcd_out_table = abcd_identity.copy()
    else:
        abcd_out_table = np.asarray(abcd_out, dtype=np.complex128)
    for name, tab in (("abcd_in", abcd_in_table), ("abcd_out", abcd_out_table)):
        if tab.shape != (n_freq, n_modes, 4):
            raise ValueError(
                f"{name} must have shape (n_freq, n_modes, 4)="
                f"({n_freq}, {n_modes}, 4), got {tab.shape}."
            )

    # apcz = A + C*Ze  and  tof = det/(A + C*Ze)  are field-independent
    # (don't depend on the self-consistent iteration), so compute them
    # once here in plain NumPy rather than per-thread inside either njit
    # kernel's prange loop.
    def _apcz_tof(abcd_table, Ze_tab):
        A, B, C, D = (abcd_table[..., k] for k in range(4))
        apcz = A + C * Ze_tab
        tof = (A * D - B * C) / apcz
        return apcz, tof

    apcz_in_table, tof_in_table = _apcz_tof(abcd_in_table, Ze_table)
    apcz_out_table, tof_out_table = _apcz_tof(abcd_out_table, Ze_table)

    if gammas_array.ndim == 3:
        if segment_midpoints is None:
            raise ValueError(
                "segment_midpoints is required for segmented devices "
                "(gammas_array.ndim == 3)."
            )
        if Zb_table.ndim != 3 or Zb_table.shape != gammas_array.shape:
            raise ValueError(
                "For segmented devices Zb_table must have shape "
                "(n_freq, n_modes, n_seg) matching gammas_array. "
                f"Got Zb_table.shape={Zb_table.shape}, "
                f"gammas_array.shape={gammas_array.shape}."
            )

        return _general_cmes_solve_fb_cutoff_segmented(
            np.ascontiguousarray(x_array),
            np.ascontiguousarray(y0_array_fwd),
            np.ascontiguousarray(y0_array_bwd),
            np.ascontiguousarray(gammas_array),
            np.ascontiguousarray(Zb_table),
            relations_3wm,
            relations_4wm,
            np.ascontiguousarray(coeffs_3wm),
            np.ascontiguousarray(coeffs_4wm),
            np.ascontiguousarray(segment_midpoints),
            passes,
            kappa_cutoff,
            update_rate,
            convergence_threshold,
            save_all_passes,
            np.ascontiguousarray(Ze_table),
            np.ascontiguousarray(Zout_in_table),
            np.ascontiguousarray(Zout_out_table),
            np.ascontiguousarray(apcz_in_table),
            np.ascontiguousarray(apcz_out_table),
            np.ascontiguousarray(tof_in_table),
            np.ascontiguousarray(tof_out_table),
        )
    else:
        if Zb_table.ndim != 2 or Zb_table.shape != gammas_array.shape:
            raise ValueError(
                "For uniform devices Zb_table must have shape "
                "(n_freq, n_modes) matching gammas_array. "
                f"Got Zb_table.shape={Zb_table.shape}, "
                f"gammas_array.shape={gammas_array.shape}."
            )
        return _general_cmes_solve_fb_cutoff_uniform(
            np.ascontiguousarray(x_array),
            np.ascontiguousarray(y0_array_fwd),
            np.ascontiguousarray(y0_array_bwd),
            np.ascontiguousarray(gammas_array),
            np.ascontiguousarray(Zb_table),
            relations_3wm,
            relations_4wm,
            np.ascontiguousarray(coeffs_3wm),
            np.ascontiguousarray(coeffs_4wm),
            passes,
            kappa_cutoff,
            update_rate,
            convergence_threshold,
            save_all_passes,
            np.ascontiguousarray(Ze_table),
            np.ascontiguousarray(Zout_in_table),
            np.ascontiguousarray(Zout_out_table),
            np.ascontiguousarray(apcz_in_table),
            np.ascontiguousarray(apcz_out_table),
            np.ascontiguousarray(tof_in_table),
            np.ascontiguousarray(tof_out_table),
        )
