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
from twpasolver.cmes.solver_config import (
    BOUNDARY_CORR_MAX,
    SOLVER_ATOL,
    SOLVER_FIRST_STEP,
    SOLVER_MAX_STEPS,
    SOLVER_RK_METHOD,
    SOLVER_RTOL,
)


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


@nb.njit(cache=True)
def _filter_wm_terms(kappas, relations_3wm, relations_4wm, relative_deltak_cutoff):
    """
    Filter phase-mismatched wave-mixing terms and expand sign combinations.

    Each base relation in ``relations_3wm``/``relations_4wm`` is expanded into
    up to 4/8 sign variants (one per +/- combination of the input modes), and
    only the phase-matched variants are kept. Because this both drops and
    duplicates rows, the output row index no longer corresponds to a row of
    ``coeffs_3wm``/``coeffs_4wm``. The last column of the returned arrays
    therefore carries the *original* relation index (the row into
    ``coeffs_3wm``/``coeffs_4wm``), so callers must index coefficients with
    that column rather than the target-mode column.
    """
    num_modes = len(kappas)
    # Extra trailing column holds the coefficient row index (into
    # coeffs_3wm/coeffs_4wm), separate from column 0 (target mode index).
    valid_3wm = np.empty((len(relations_3wm) * 4, 4), dtype=relations_3wm.dtype)
    valid_4wm = np.empty((len(relations_4wm) * 8, 5), dtype=relations_4wm.dtype)
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
                valid_3wm[nv3, 3] = k3
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
                valid_4wm[nv4, 4] = k4
                nv4 += 1

    return valid_3wm[:nv3], valid_4wm[:nv4]


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
    CME RHS for uniform-device forward and backward passes.

    currents_evals holds the *native* (unreversed) slow envelope B_m of the
    counter-propagating solve, sampled on its own grid x_evals. Since both
    forward and backward solves share the same coordinate x, the field of
    the counter-propagating wave at physical position x lives at grid
    coordinate (end - x) in currents_evals: no reversed copy is needed, and
    this is exact for any x grid (not just a mirror-symmetric one).
    When currents_evals is all zeros (first forward pass) the counter term
    vanishes, reproducing the no-backward-field equation exactly.

    Phase carriers:
      forward  carrier: ep[m] = exp(i * kappas[m] * x)
      backward carrier: en[m] = exp(i * kappas[m] * (end - x))

    Returns dB_m/dx (full, with the +alpha*B term), length n_modes. The raw
    nonlinear-only part used for the reflection-coefficient update is not
    captured here; it's cheap to recover as (this output - alpha*currents)
    from a direct call at the two endpoints where it's actually needed.
    """
    n = len(currents)
    out = np.zeros(n, nb.complex128)
    ep = np.exp(1j * kappas * x)
    en = np.exp(1j * kappas * (end - x))

    ar = np.empty(4 * n, dtype=nb.complex128)
    # x_evals is sorted: one binary search gives the bracket for all modes.
    x_ctr = end - x
    lo_e, hi_e = _sorted_interp_bracket(x_evals, x_ctr)
    frac_e = (
        0.0
        if lo_e == hi_e
        else (x_ctr - x_evals[lo_e]) / (x_evals[hi_e] - x_evals[lo_e])
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

    # relations_3wm/4wm carry a trailing coefficient-row index (see
    # _filter_wm_terms): column 0 is the target mode, the last column
    # indexes coeffs_3wm/coeffs_4wm -- these are NOT the same in general.
    for idx, i1, i2, cidx in relations_3wm:
        out[idx] += coeffs_3wm[cidx] * ar[i1] * ar[i2]
    for idx, i1, i2, i3, cidx in relations_4wm:
        out[idx] += coeffs_4wm[cidx] * ar[i1] * ar[i2] * ar[i3]

    for i in range(n):
        raw = (alphas[i] + 1j * kappas[i]) * out[i] / ep[i]
        out[i] = raw + alphas[i] * currents[i]

    return out


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
        nb_float1d,
        nb.float64,
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
    counter_x_evals,
    end,
):
    """
    CME RHS for segmented-device forward and backward passes.

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
        Sorted grid on which ``phi_table`` is defined -- this solve's own
        coordinate grid.
    phi_table : ndarray, shape (n_modes, n_x)
        Cumulative phase per mode, precomputed on ``x_evals`` (own grid).
    currents_evals : ndarray, shape (n_modes, n_x_ctr)
        *Native* (unreversed) slow envelope of the counter-propagating
        solve, sampled on its own grid ``counter_x_evals``. Zero on the
        first forward pass, so the counter term vanishes.
    counter_x_evals : ndarray, shape (n_x_ctr,)
        Sorted coordinate grid on which ``currents_evals`` is defined.
    end : float
        Physical device length. The counter-propagating field at physical
        position x is read off currents_evals at coordinate (end - x) on
        its own grid, so no reversed copy is needed and this is exact for
        any (not necessarily mirror-symmetric) grid.

    Returns
    -------
    ndarray
        ``dB_m/dx``, length n_modes (full, with the attenuation term). The
        raw nonlinear-only part used for the reflection-coefficient update
        is not captured here; recover it as (this output - alpha*currents)
        from a direct call at the two endpoints where it's actually needed.
    """
    n = len(currents)
    out = np.zeros(n, nb.complex128)

    # locate x in the parameter-table grid and own-position grid
    lo_s, hi_s = _sorted_interp_bracket(segment_midpoints, x)
    lo_x, hi_x = _sorted_interp_bracket(x_evals, x)
    # locate the counter-propagating coordinate (end - x) in its own grid
    x_ctr = end - x
    lo_c, hi_c = _sorted_interp_bracket(counter_x_evals, x_ctr)

    frac_s = (
        0.0
        if lo_s == hi_s
        else (x - segment_midpoints[lo_s])
        / (segment_midpoints[hi_s] - segment_midpoints[lo_s])
    )
    frac_x = (
        0.0 if lo_x == hi_x else (x - x_evals[lo_x]) / (x_evals[hi_x] - x_evals[lo_x])
    )
    frac_c = (
        0.0
        if lo_c == hi_c
        else (x_ctr - counter_x_evals[lo_c])
        / (counter_x_evals[hi_c] - counter_x_evals[lo_c])
    )

    # prepare carriers and interpolated coefficients
    alphas = np.empty(n, dtype=nb.float64)
    kappas = np.empty(n, dtype=nb.float64)
    ep = np.empty(n, dtype=nb.complex128)  # forward carrier
    en = np.empty(n, dtype=nb.complex128)  # backward carrier
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
        en[m] = np.exp(1j * (phi_total - phi_x))

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

    # assemble field amplitudes
    ar = np.empty(4 * n, dtype=nb.complex128)
    for i in range(n):
        ctr_re = currents_evals[i, lo_c].real + frac_c * (
            currents_evals[i, hi_c].real - currents_evals[i, lo_c].real
        )
        ctr_im = currents_evals[i, lo_c].imag + frac_c * (
            currents_evals[i, hi_c].imag - currents_evals[i, lo_c].imag
        )
        ctr = (ctr_re + 1j * ctr_im) * en[i]
        ar[i] = currents[i] * ep[i]  # co-propagating with carrier
        ar[i + n] = np.conj(ar[i])
        ar[i + 2 * n] = ctr  # counter-propagating with carrier
        ar[i + 3 * n] = np.conj(ctr)

    # nonlinear interactions. relations_3wm/4wm carry a trailing
    # coefficient-row index (see _filter_wm_terms): column 0 is the target
    # mode, the last column indexes coeffs_3wm/coeffs_4wm -- these are NOT
    # the same in general.
    for idx, i1, i2, cidx in relations_3wm:
        out[idx] += coeffs_3wm[cidx] * ar[i1] * ar[i2]
    for idx, i1, i2, i3, cidx in relations_4wm:
        out[idx] += coeffs_4wm[cidx] * ar[i1] * ar[i2] * ar[i3]

    # loss and derivatives
    for i in range(n):
        raw = (alphas[i] + 1j * kappas[i]) * out[i] / ep[i]
        out[i] = raw + alphas[i] * currents[i]

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
    valid only while |raw_launch / (gamma * B_launch)| stays small (SVEA).
    Close to a stopband edge, nonlinear interactions can push this ratio to O(1)
    or larger and can drive refl_new outside the physical |Gamma| <= 1 regime.
    Two safeguards are applied:
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
    raw_fwd_end,
    raw_bwd_0,
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
    """Self-consistent reflection/transmission update at the output (x=L) boundary.

    ``raw_fwd_end``/``raw_bwd_0`` are the raw nonlinear-only RHS terms at the
    two relevant endpoints (forward at x=L, backward at x=0), recovered by
    the caller with a direct RHS call rather than captured during the solve.
    """
    for j in range(n_modes):
        refl_end_out[j], trans_end_in[j] = _boundary_reflection_update(
            raw_bwd_0[j],
            A_bwd_i[j, 0],
            raw_fwd_end[j],
            A_fwd_i[j, -1],
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
    raw_fwd_0,
    raw_bwd_end,
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
    ``raw_fwd_0``/``raw_bwd_end`` are the raw nonlinear-only RHS terms at the
    two relevant endpoints (forward at x=0, backward at x=L), recovered by
    the caller with a direct RHS call rather than captured during the solve.
    """
    for j in range(n_modes):
        if do_update:
            refl_0_out[j], trans_0_in[j] = _boundary_reflection_update(
                raw_fwd_0[j],
                A_fwd_i[j, 0],
                raw_bwd_end[j],
                A_bwd_i[j, -1],
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
    n_fwd = 0.0
    n_bwd = 0.0
    for j in range(n_modes):
        v = np.abs(A_fwd_i[j, -1] - prev_fwd[j]) / (np.abs(A_fwd_i[j, -1]) + 1e-30)
        w = np.abs(A_bwd_i[j, -1] - prev_bwd[j]) / (np.abs(A_bwd_i[j, -1]) + 1e-30)
        if v > n_fwd:
            n_fwd = v
        if w > n_bwd:
            n_bwd = w
    if n_fwd < convergence_threshold and n_bwd < convergence_threshold:
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
    I_out[idx_fwd] = A_fwd_i
    I_out[idx_bwd] = A_bwd_i
    for j in range(n_modes):
        I_out[idx_fwd, j, -1] = A_fwd_i[j, -1] * (1.0 + refl_end_out[j]) * tof_out[j]
        I_out[idx_bwd, j, -1] = A_bwd_i[j, -1] * (1.0 + refl_0_out[j]) * tof_in[j]


@nb.njit(parallel=True, cache=True)
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
    n_x = len(x_array)
    x_end = x_array[-1]
    x_span = (x_array[0], x_array[-1])
    x_array0 = np.ascontiguousarray(x_array)

    if save_all_passes:
        I_out = np.empty((2 * n_freq * passes, n_modes, n_x), dtype=np.complex128)
    else:
        I_out = np.empty((2 * n_freq, n_modes, n_x), dtype=np.complex128)

    for i in nb.prange(n_freq):
        I_fwd = np.zeros(n_modes, dtype=np.complex128)
        I_bwd = np.zeros(n_modes, dtype=np.complex128)
        refl_end_out = np.zeros(n_modes, dtype=np.complex128)
        refl_0_out = np.zeros(n_modes, dtype=np.complex128)
        # Counter field for the very first forward solve: no backward wave yet.
        A_bwd_i = np.zeros((n_modes, n_x), dtype=np.complex128)

        converged = False
        stable_count = 0
        prev_fwd = np.zeros(n_modes, dtype=np.complex128)
        prev_bwd = np.zeros(n_modes, dtype=np.complex128)

        alphas_i = np.ascontiguousarray(gammas_array[i].real)
        kappas_i = np.ascontiguousarray(gammas_array[i].imag)
        Zb_i = Zb_table[i]
        gam_i = gammas_array[i]

        # boundary two-port tables for this frequency
        Ze_i = Ze_table[i]
        Zout_in_i = Zout_in_table[i]
        Zout_out_i = Zout_out_table[i]
        apcz_in = apcz_in_table[i]
        apcz_out = apcz_out_table[i]
        tof_in = tof_in_table[i]
        tof_out = tof_out_table[i]

        # Full-phase-at-x_end carrier, shared by both boundary launches.
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
            # forward solve. Counter field is A_bwd_i (previous pass's
            # backward solution, or all-zero on the very first pass).
            A_fwd_i = nbsolve_ivp(
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
                    x_array0,
                    A_bwd_i,
                    x_end,
                ),
                atol=SOLVER_ATOL,
                rtol=SOLVER_RTOL,
                max_num_steps=SOLVER_MAX_STEPS,
                t_eval=x_array0,
                rk_method=SOLVER_RK_METHOD,
                first_step=SOLVER_FIRST_STEP,
            ).y

            # Raw nonlinear-only RHS at the two endpoints the boundary
            # updates need, recovered with a direct call using the same
            # counter state (A_bwd_i) this solve itself used -- cheaper
            # than capturing and dense-interpolating it for the whole grid.
            raw_fwd_0 = (
                cme_rhs(
                    x_array0[0],
                    A_fwd_i[:, 0],
                    alphas_i,
                    kappas_i,
                    r3,
                    r4,
                    coeffs_3wm,
                    coeffs_4wm,
                    x_array0,
                    A_bwd_i,
                    x_end,
                )
                - alphas_i * A_fwd_i[:, 0]
            )
            raw_fwd_end = (
                cme_rhs(
                    x_array0[-1],
                    A_fwd_i[:, -1],
                    alphas_i,
                    kappas_i,
                    r3,
                    r4,
                    coeffs_3wm,
                    coeffs_4wm,
                    x_array0,
                    A_bwd_i,
                    x_end,
                )
                - alphas_i * A_fwd_i[:, -1]
            )

            # reflection at far end updates backward initial conditions
            _compute_backward_launch(
                A_fwd_i,
                y0_array_bwd[i],
                trans_end_in,
                refl_end_out,
                phase_end,
                I_bwd,
                n_modes,
            )

            # backward solve. Counter field is A_fwd_i, just computed above.
            A_bwd_i = nbsolve_ivp(
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
                    x_array0,
                    A_fwd_i,
                    x_end,
                ),
                atol=SOLVER_ATOL,
                rtol=SOLVER_RTOL,
                max_num_steps=SOLVER_MAX_STEPS,
                t_eval=x_array0,
                rk_method=SOLVER_RK_METHOD,
                first_step=SOLVER_FIRST_STEP,
            ).y

            raw_bwd_0 = (
                cme_rhs(
                    x_array0[0],
                    A_bwd_i[:, 0],
                    alphas_i,
                    kappas_i,
                    r3,
                    r4,
                    coeffs_3wm,
                    coeffs_4wm,
                    x_array0,
                    A_fwd_i,
                    x_end,
                )
                - alphas_i * A_bwd_i[:, 0]
            )
            raw_bwd_end = (
                cme_rhs(
                    x_array0[-1],
                    A_bwd_i[:, -1],
                    alphas_i,
                    kappas_i,
                    r3,
                    r4,
                    coeffs_3wm,
                    coeffs_4wm,
                    x_array0,
                    A_fwd_i,
                    x_end,
                )
                - alphas_i * A_bwd_i[:, -1]
            )

            # launch = backward wave (into TWPA), inc = forward wave (arriving).
            if k > 1:
                _update_output_boundary(
                    A_fwd_i,
                    A_bwd_i,
                    raw_fwd_end,
                    raw_bwd_0,
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

            # launch = forward wave (into TWPA), inc = backward wave (arriving).
            _update_input_boundary_and_launch(
                A_fwd_i,
                A_bwd_i,
                raw_fwd_0,
                raw_bwd_end,
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

            # convergence check
            if k > 1:
                stable_count, converged = _convergence_step(
                    A_fwd_i,
                    A_bwd_i,
                    prev_fwd,
                    prev_bwd,
                    n_modes,
                    convergence_threshold,
                    stable_count,
                )

            # store output
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
                A_fwd_i,
                A_bwd_i,
                refl_end_out,
                refl_0_out,
                tof_out,
                tof_in,
                i + id_add_fwd,
                i + id_add_bwd,
                n_modes,
            )

            for j in range(n_modes):
                prev_fwd[j] = A_fwd_i[j, -1]
                prev_bwd[j] = A_bwd_i[j, -1]

            # early exit on convergence
            if converged:
                if save_all_passes:
                    for kk in range(k + 1, passes):
                        _store_pass(
                            I_out,
                            A_fwd_i,
                            A_bwd_i,
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


@nb.njit(parallel=True, cache=True)
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

    where ``raw`` is the pure nonlinear derivative (recovered from a direct
    RHS call at the relevant endpoint, see below) and ``B`` the slow
    envelope.  With  W = (A + C*Ze)*Ztilde_launch + (B + D*Ze),
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

    Counter-propagating field
    --------------------------
    Each solve passes the *other* direction's raw (native, unreversed) slow
    envelope as the counter field, together with the grid it's defined on.
    Inside cme_rhs_segmented the counter field at physical position x is
    read off that array at coordinate (x_end - x) on its own grid -- exact
    for any grid, without needing a mirror-symmetric one or a reversed copy.
    """
    n_freq = gammas_array.shape[0]
    n_modes = gammas_array.shape[1]
    n_x = len(x_array)
    x_span = (x_array[0], x_array[-1])
    x_array0 = np.ascontiguousarray(x_array)

    if save_all_passes:
        I_out = np.empty((2 * n_freq * passes, n_modes, n_x), dtype=np.complex128)
    else:
        I_out = np.empty((2 * n_freq, n_modes, n_x), dtype=np.complex128)

    for i in nb.prange(n_freq):
        alphas_i = np.ascontiguousarray(gammas_array[i].real)  # (n_modes, n_seg)
        kappas_i = np.ascontiguousarray(gammas_array[i].imag)  # (n_modes, n_seg)

        # boundary two-port tables for this frequency
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

        # Use average kappas for the wave-mixing filter (conservative: keeps
        # terms that are phase-matched anywhere in the device).
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

        atol_fwd = np.full(n_modes, SOLVER_ATOL)
        atol_bwd = np.full(n_modes, SOLVER_ATOL)

        # Counter field for the very first forward solve: no backward wave
        # yet. Defined on the backward solve's own (mirrored) grid.
        A_bwd_i = np.zeros((n_modes, n_x), dtype=np.complex128)

        for k in range(passes):
            # forward solve. Counter field is A_bwd_i (previous pass's
            # backward solution on its own s-grid x_array_bwd, or all-zero
            # on the very first pass).
            A_fwd_i = nbsolve_ivp(
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
                    x_array0,
                    phi_table,
                    A_bwd_i,
                    x_array_bwd,
                    x_end_val,
                ),
                atols=atol_fwd,
                rtol=SOLVER_RTOL,
                max_num_steps=SOLVER_MAX_STEPS,
                t_eval=x_array0,
                rk_method=SOLVER_RK_METHOD,
                first_step=SOLVER_FIRST_STEP,
            ).y

            # Raw nonlinear-only RHS at the two endpoints the boundary
            # updates need, recovered with a direct call using the same
            # counter state (A_bwd_i) this solve itself used. Physical
            # x=0 -> alpha=gam_0.real, physical x=x_end -> alpha=gam_end.real
            # (segment tables clamp at the domain edges, matching Zb_0/gam_0
            # and Zb_end/gam_end used for the boundary conditions).
            raw_fwd_0 = (
                cme_rhs_segmented(
                    x_array0[0],
                    A_fwd_i[:, 0],
                    alphas_i,
                    kappas_i,
                    r3,
                    r4,
                    coeffs_3wm_table,
                    coeffs_4wm_table,
                    segment_midpoints,
                    x_array0,
                    phi_table,
                    A_bwd_i,
                    x_array_bwd,
                    x_end_val,
                )
                - gam_0.real * A_fwd_i[:, 0]
            )
            raw_fwd_end = (
                cme_rhs_segmented(
                    x_array0[-1],
                    A_fwd_i[:, -1],
                    alphas_i,
                    kappas_i,
                    r3,
                    r4,
                    coeffs_3wm_table,
                    coeffs_4wm_table,
                    segment_midpoints,
                    x_array0,
                    phi_table,
                    A_bwd_i,
                    x_array_bwd,
                    x_end_val,
                )
                - gam_end.real * A_fwd_i[:, -1]
            )

            # reflection at far end updates backward initial conditions
            _compute_backward_launch(
                A_fwd_i,
                y0_array_bwd[i],
                trans_end_in,
                refl_end_out,
                phase_end,
                I_bwd,
                n_modes,
            )

            # backward solve. Counter field is A_fwd_i, just computed above,
            # on its own grid x_array0. All parameter tables are pre-reversed
            # (segment axis flipped) so the RHS reads physical positions
            # from the output end first as the solver steps from s=0 to
            # s=x_end.
            A_bwd_i = nbsolve_ivp(
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
                    A_fwd_i,
                    x_array0,
                    x_end_val,
                ),
                atols=atol_bwd,
                rtol=SOLVER_RTOL,
                max_num_steps=SOLVER_MAX_STEPS,
                t_eval=x_array_bwd,
                rk_method=SOLVER_RK_METHOD,
                first_step=SOLVER_FIRST_STEP,
            ).y

            # s=0 -> physical x_end -> alpha=gam_end.real;
            # s=x_end -> physical x=0 -> alpha=gam_0.real.
            raw_bwd_0 = (
                cme_rhs_segmented(
                    x_array_bwd[0],
                    A_bwd_i[:, 0],
                    alphas_i_bwd,
                    kappas_i_bwd,
                    r3,
                    r4,
                    coeffs_3wm_bwd,
                    coeffs_4wm_bwd,
                    segment_midpoints_bwd,
                    x_array_bwd,
                    phi_table_bwd,
                    A_fwd_i,
                    x_array0,
                    x_end_val,
                )
                - gam_end.real * A_bwd_i[:, 0]
            )
            raw_bwd_end = (
                cme_rhs_segmented(
                    x_array_bwd[-1],
                    A_bwd_i[:, -1],
                    alphas_i_bwd,
                    kappas_i_bwd,
                    r3,
                    r4,
                    coeffs_3wm_bwd,
                    coeffs_4wm_bwd,
                    segment_midpoints_bwd,
                    x_array_bwd,
                    phi_table_bwd,
                    A_fwd_i,
                    x_array0,
                    x_end_val,
                )
                - gam_0.real * A_bwd_i[:, -1]
            )

            # launch = backward wave (into TWPA), inc = forward wave (arriving).
            if k > 1:
                _update_output_boundary(
                    A_fwd_i,
                    A_bwd_i,
                    raw_fwd_end,
                    raw_bwd_0,
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

            # launch = forward wave (into TWPA), inc = backward wave (arriving).
            _update_input_boundary_and_launch(
                A_fwd_i,
                A_bwd_i,
                raw_fwd_0,
                raw_bwd_end,
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

            # update per-mode atol vectors after the first pass
            if k == 0:
                atol_fwd = _compute_atol_vec(A_fwd_i, n_modes, SOLVER_ATOL)
                atol_bwd = _compute_atol_vec(A_bwd_i, n_modes, SOLVER_ATOL)

            # convergence check
            if k > 1:
                stable_count, converged = _convergence_step(
                    A_fwd_i,
                    A_bwd_i,
                    prev_fwd,
                    prev_bwd,
                    n_modes,
                    convergence_threshold,
                    stable_count,
                )

            # store output
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
                A_fwd_i,
                A_bwd_i,
                refl_end_out,
                refl_0_out,
                tof_out,
                tof_in,
                i + id_add_fwd,
                i + id_add_bwd,
                n_modes,
            )

            for j in range(n_modes):
                prev_fwd[j] = A_fwd_i[j, -1]
                prev_bwd[j] = A_bwd_i[j, -1]

            # early exit on convergence
            if converged:
                if save_all_passes:
                    for kk in range(k + 1, passes):
                        _store_pass(
                            I_out,
                            A_fwd_i,
                            A_bwd_i,
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

    # apcz = A + C*Ze  and  tof = det/(A + C*Ze)  are field-independent
    # (don't depend on the self-consistent iteration), so compute them
    # once here rather than per-thread inside njit. For identity ABCD
    # (A=D=1, B=C=0), apcz = 1 and tof = 1 identically -- skip building the
    # (n_freq, n_modes, 4) array entirely in that (default) case.
    ones_table = np.ones((n_freq, n_modes), dtype=np.complex128)

    def _apcz_tof(abcd, Ze_tab):
        if abcd is None:
            return ones_table, ones_table
        tab = np.asarray(abcd, dtype=np.complex128)
        if tab.shape != (n_freq, n_modes, 4):
            raise ValueError(
                f"abcd table must have shape (n_freq, n_modes, 4)="
                f"({n_freq}, {n_modes}, 4), got {tab.shape}."
            )
        A, B, C, D = (tab[..., k] for k in range(4))
        apcz = A + C * Ze_tab
        tof = (A * D - B * C) / apcz
        return apcz, tof

    apcz_in_table, tof_in_table = _apcz_tof(abcd_in, Ze_table)
    apcz_out_table, tof_out_table = _apcz_tof(abcd_out, Ze_table)

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
