"""
Forward-backward Bloch CMEs for segmented (piecewise-uniform) devices.

A segmented device is a chain of uniform segments, each made of whole
supercells of its own. Inside segment ``s`` the equations are exactly those of
:mod:`twpasolver.cmes.bloch_fb`, written on that segment's Bloch modes
(``gamma_s``, ``Zb±_s``, ``W_s``, intracell profiles, reciprocal vectors
``2 pi m / d_s``, term selection and stopband-edge weights). Segments are joined
by exact algebraic junction conditions.

Junction conditions
-------------------
Every junction lies on a supercell boundary of both neighbours, where the
expansion ``psi = A+ psi+ + A- psi-`` gives ``I = A+ + A-`` and
``V = Zb+ A+ - Zb- A-`` exactly (``phi = 1`` at supercell boundaries), with the
nonlinearity on. Continuity of (V, I) and the projection with the reciprocity
invariant give, for the junction between a left segment L and a right
segment R,

    A+_R = t  A+_L + rho_R A-_R,      t   = W_L / (Zb+_R + Zb-_L),
    A-_L = t' A-_R + rho_L A+_L,      t'  = W_R / (Zb+_R + Zb-_L),
                                      rho_R = (Zb-_R - Zb-_L) / (Zb+_R + Zb-_L),
                                      rho_L = (Zb+_L - Zb+_R) / (Zb+_R + Zb-_L).

The input boundary of :mod:`bloch_fb` is the special case ``Zb-_L -> Zout``.
Identical neighbours give ``t = t' = 1``, ``rho = 0``. In flux variables
(``a ∝ sqrt(W) A``) the junction map is reciprocal, and unitary for lossless
passbands, so energy and Manley-Rowe conservation survive composition.

Phases and coordinates
----------------------
As in the uniform kernel only the propagation phase is factored out of the
integration variables. With the piecewise-constant wavenumber the phase is

    Phi_f(x) = integral_0^x kappa(x') dx'         (forward, physical x),
    Phi_b(s) = Phi_f(L) - Phi_f(L - s)            (backward, mirrored s = L - x),

and the fields entering the products are ``B+ e^{j Phi_f}``, ``B- e^{j Phi_b}``.
Reciprocal-vector phases are referred to the origin of each segment
(``exp(j G_m (x - x_s))`` in the solve coordinate), which is exact in both
directions because every segment holds a whole number of its supercells.

Grid
----
Envelopes are discontinuous at junctions. The solver works on an internal grid
made of one sub-grid per segment, each containing both segment ends (junction
positions appear twice: left and right limits). Counter-propagating fields are
interpolated only inside the segment being integrated, and the integrator is
restarted at every junction. Results are returned on the caller's grid
(the left limit is used at a junction position).

Iteration
---------
One pass is a forward sweep over the segments (junction steps use the current
backward estimate) followed by a backward sweep (junction steps use the
forward fields just computed). The pass map acts on the backward envelopes on
the internal grid; Anderson acceleration is unchanged from :mod:`bloch_fb`.
Internal reflections add round trips to the pass map, which Anderson
acceleration handles like the end reflections.
"""

from __future__ import annotations

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
from twpasolver.cmes import solver_config
from twpasolver.cmes.base import _sorted_interp_bracket
from twpasolver.cmes.bloch_fb import (
    _anderson_mix,
    _bloch_row_coeffs,
    _filter_wm_terms_bloch,
    _mode_scales,
    _relative_residual,
)
from twpasolver.cmes.cmes_fb import _convergence_step, _store_pass
from twpasolver.cmes.rk45 import rk45_solve
from twpasolver.cmes.solver_config import (
    SOLVER_ATOL,
    SOLVER_FIRST_STEP,
    SOLVER_MAX_STEPS,
    SOLVER_RTOL,
)
from twpasolver.logger import log

# --------------------------------------------------------------------------- #
# Linear pieces (plain NumPy)
# --------------------------------------------------------------------------- #


def junction_coefficients(
    Zb_plus: ComplexArray, Zb_minus: ComplexArray
) -> tuple[ComplexArray, ComplexArray, ComplexArray, ComplexArray]:
    """
    Exact junction coefficients between consecutive segments.

    Parameters
    ----------
    Zb_plus, Zb_minus : (..., n_seg, n_modes) complex
        Bloch impedances of every segment (forward / backward mode), referred
        to the supercell boundary that sits at the junction.

    Returns
    -------
    t, t_prime, rho_L, rho_R : (..., n_seg, n_modes) complex
        Entry ``j`` describes the junction between segments ``j - 1`` (left)
        and ``j`` (right). Entry 0 is the identity (no junction).
    """
    Zp = np.asarray(Zb_plus, dtype=np.complex128)
    Zm = np.asarray(Zb_minus, dtype=np.complex128)
    t = np.ones_like(Zp)
    tp = np.ones_like(Zp)
    rho_L = np.zeros_like(Zp)
    rho_R = np.zeros_like(Zp)
    ZpL, ZmL = Zp[..., :-1, :], Zm[..., :-1, :]
    ZpR, ZmR = Zp[..., 1:, :], Zm[..., 1:, :]
    den = ZpR + ZmL
    t[..., 1:, :] = (ZpL + ZmL) / den
    tp[..., 1:, :] = (ZpR + ZmR) / den
    rho_R[..., 1:, :] = (ZmR - ZmL) / den
    rho_L[..., 1:, :] = (ZpL - ZpR) / den
    return t, tp, rho_L, rho_R


def build_segment_grid(
    x_array: FloatArray, seg_bounds: FloatArray
) -> tuple[FloatArray, np.ndarray, np.ndarray]:
    """
    Internal grid with both ends of every segment, and the map to ``x_array``.

    Returns
    -------
    x_int : (n_int,) float
        Concatenation of the per-segment grids ``[x_s, interior x_array points,
        x_{s+1}]``; junction positions appear twice.
    offsets : (n_seg + 1,) int
        Segment ``s`` occupies ``x_int[offsets[s]:offsets[s+1]]``.
    take : (n_x,) int
        Index into ``x_int`` of the value reported at each ``x_array`` point
        (left limit at a junction, segment 0 at ``x = 0``).
    """
    x_array = np.asarray(x_array, dtype=np.float64)
    seg_bounds = np.asarray(seg_bounds, dtype=np.float64)
    n_seg = len(seg_bounds) - 1
    pieces = []
    offsets = np.zeros(n_seg + 1, dtype=np.int64)
    for s in range(n_seg):
        lo, hi = seg_bounds[s], seg_bounds[s + 1]
        inner = x_array[(x_array > lo) & (x_array < hi)]
        pieces.append(np.concatenate([[lo], inner, [hi]]))
        offsets[s + 1] = offsets[s] + len(pieces[-1])
    x_int = np.concatenate(pieces)

    take = np.empty(len(x_array), dtype=np.int64)
    for k, xv in enumerate(x_array):
        s = int(np.searchsorted(seg_bounds, xv, side="left")) - 1
        s = min(max(s, 0), n_seg - 1)
        sub = x_int[offsets[s] : offsets[s + 1]]
        take[k] = offsets[s] + int(np.argmin(np.abs(sub - xv)))
    return x_int, offsets, take


def segmented_phase(kappas: FloatArray, seg_bounds: FloatArray, x: FloatArray):
    """
    Exact cumulative propagation phase for piecewise-constant wavenumbers.

    Parameters
    ----------
    kappas : (..., n_seg) float
        Kernel wavenumbers (``Im`` of the kernel gammas) of every segment.
    seg_bounds : (n_seg + 1,) float
    x : (n_x,) float

    Returns
    -------
    (Phi_f, Phi_b) : (..., n_x) float
        Forward phase at physical ``x`` and backward phase at mirrored
        coordinate ``s = x`` (i.e. physical position ``L - x``).
    """
    kappas = np.asarray(kappas, dtype=np.float64)
    seg_bounds = np.asarray(seg_bounds, dtype=np.float64)
    L = seg_bounds[-1]
    lengths = np.diff(seg_bounds)
    cum = np.concatenate(
        [np.zeros(kappas.shape[:-1] + (1,)), np.cumsum(kappas * lengths, axis=-1)],
        axis=-1,
    )

    def phi(xv):
        s = np.clip(
            np.searchsorted(seg_bounds, xv, side="right") - 1, 0, len(lengths) - 1
        )
        return cum[..., s] + kappas[..., s] * (xv - seg_bounds[s])

    x = np.asarray(x, dtype=np.float64)
    Phi_f = phi(x)
    Phi_b = cum[..., -1:] - phi(L - x)
    return Phi_f, Phi_b


# --------------------------------------------------------------------------- #
# Kernel
# --------------------------------------------------------------------------- #


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
        nb_float1d,
        nb.float64,
        nb_float1d,
        nb_float1d,
        nb_complex2d,
        nb.float64,
        nb_float1d,
        nb.float64,
    ),
    cache=True,
)
def bloch_fb_segment_rhs(
    x,
    currents,
    alphas,
    kappas,
    rows_3wm,
    rows_4wm,
    row_coeffs_3wm,
    row_coeffs_4wm,
    row_g_3wm,
    row_g_4wm,
    x0,
    phi0,
    ctr_x,
    ctr_vals,
    ctr_y0,
    ctr_phi0,
    end,
):
    """
    RHS of the Bloch forward/backward CMEs inside one segment.

    Same field bookkeeping as :func:`twpasolver.cmes.bloch_fb.bloch_fb_rhs`,
    with segment-local phases:

    * own carrier ``exp(j (phi0 + kappa (x - x0)))``, ``x0`` the start of the
      segment in the solve coordinate and ``phi0`` the accumulated phase there;
    * counter field read at ``y = end - x`` from ``ctr_vals`` sampled on
      ``ctr_x`` (the same physical segment in the counter coordinate, whose
      start is ``ctr_y0`` with phase ``ctr_phi0``);
    * reciprocal-vector phase ``exp(j G (x - x0))``.
    """
    n = len(currents)
    out = np.zeros(n, nb.complex128)
    xl = x - x0
    y = end - x
    yl = y - ctr_y0
    ep = np.exp(1j * (phi0 + kappas * xl))
    en = np.exp(1j * (ctr_phi0 + kappas * yl))

    ar = np.empty(4 * n, dtype=nb.complex128)
    lo_e, hi_e = _sorted_interp_bracket(ctr_x, y)
    frac_e = 0.0 if lo_e == hi_e else (y - ctr_x[lo_e]) / (ctr_x[hi_e] - ctr_x[lo_e])
    for i in range(n):
        ctr_re = ctr_vals[i, lo_e].real + frac_e * (
            ctr_vals[i, hi_e].real - ctr_vals[i, lo_e].real
        )
        ctr_im = ctr_vals[i, lo_e].imag + frac_e * (
            ctr_vals[i, hi_e].imag - ctr_vals[i, lo_e].imag
        )
        ctr = (ctr_re + 1j * ctr_im) * en[i]
        ar[i] = currents[i] * ep[i]
        ar[i + n] = np.conj(ar[i])
        ar[i + 2 * n] = ctr
        ar[i + 3 * n] = np.conj(ctr)

    for r in range(rows_3wm.shape[0]):
        term = row_coeffs_3wm[r] * ar[rows_3wm[r, 1]] * ar[rows_3wm[r, 2]]
        if row_g_3wm[r] != 0.0:
            term *= np.exp(1j * row_g_3wm[r] * xl)
        out[rows_3wm[r, 0]] += term
    for r in range(rows_4wm.shape[0]):
        term = (
            row_coeffs_4wm[r]
            * ar[rows_4wm[r, 1]]
            * ar[rows_4wm[r, 2]]
            * ar[rows_4wm[r, 3]]
        )
        if row_g_4wm[r] != 0.0:
            term *= np.exp(1j * row_g_4wm[r] * xl)
        out[rows_4wm[r, 0]] += term

    for i in range(n):
        out[i] = out[i] / ep[i] + alphas[i] * currents[i]
    return out


@nb.njit(cache=True)
def _segment_rows(
    kappas_seg,
    relations_3wm,
    relations_4wm,
    base_3wm,
    base_4wm,
    prefac_seg,
    phi_plus_seg,
    phi_minus_seg,
    L_nl,
    positions,
    n_cells,
    pw3_seg,
    pw4_seg,
    kappa_cutoff,
    cell_phase_cutoff,
    m_max,
):
    """
    Kept rows and their coefficients for every segment, concatenated.

    Returns ``r3, r4`` (rows), ``off3, off4`` (per-segment offsets),
    ``g3, g4`` (reciprocal vectors) and the coefficients of both solve
    directions ``c3f, c3b, c4f, c4b``.
    """
    n_seg = kappas_seg.shape[0]
    n_g = 2 * m_max + 1
    max3 = len(relations_3wm) * 4 * n_g
    max4 = len(relations_4wm) * 8 * n_g
    r3 = np.empty((n_seg * max3, 5), dtype=np.int64)
    r4 = np.empty((n_seg * max4, 6), dtype=np.int64)
    g3 = np.empty(n_seg * max3, dtype=np.float64)
    g4 = np.empty(n_seg * max4, dtype=np.float64)
    c3f = np.empty(n_seg * max3, dtype=np.complex128)
    c3b = np.empty(n_seg * max3, dtype=np.complex128)
    c4f = np.empty(n_seg * max4, dtype=np.complex128)
    c4b = np.empty(n_seg * max4, dtype=np.complex128)
    off3 = np.zeros(n_seg + 1, dtype=np.int64)
    off4 = np.zeros(n_seg + 1, dtype=np.int64)
    for s in range(n_seg):
        nc = n_cells[s]
        k_s = np.ascontiguousarray(kappas_seg[s])
        a3, a4 = _filter_wm_terms_bloch(
            k_s,
            relations_3wm,
            relations_4wm,
            kappa_cutoff,
            nc,
            cell_phase_cutoff,
            m_max,
        )
        pp = np.ascontiguousarray(phi_plus_seg[s])
        pm = np.ascontiguousarray(phi_minus_seg[s])
        pf = np.ascontiguousarray(prefac_seg[s])
        Ls = np.ascontiguousarray(L_nl[s])
        ps = np.ascontiguousarray(positions[s])
        b3 = np.ascontiguousarray(base_3wm[s])
        b4 = np.ascontiguousarray(base_4wm[s])
        cf3 = _bloch_row_coeffs(a3, b3, pf, pp, pm, Ls, ps, nc, True)
        cb3 = _bloch_row_coeffs(a3, b3, pf, pp, pm, Ls, ps, nc, False)
        cf4 = _bloch_row_coeffs(a4, b4, pf, pp, pm, Ls, ps, nc, True)
        cb4 = _bloch_row_coeffs(a4, b4, pf, pp, pm, Ls, ps, nc, False)
        g_unit = 2.0 * np.pi / nc
        o = off3[s]
        for r in range(a3.shape[0]):
            w = pw3_seg[s, a3[r, 3]]
            r3[o + r] = a3[r]
            g3[o + r] = a3[r, 4] * g_unit
            c3f[o + r] = cf3[r] * w
            c3b[o + r] = cb3[r] * w
        off3[s + 1] = o + a3.shape[0]
        o = off4[s]
        for r in range(a4.shape[0]):
            w = pw4_seg[s, a4[r, 4]]
            r4[o + r] = a4[r]
            g4[o + r] = a4[r, 5] * g_unit
            c4f[o + r] = cf4[r] * w
            c4b[o + r] = cb4[r] * w
        off4[s + 1] = o + a4.shape[0]
    n3 = off3[n_seg]
    n4 = off4[n_seg]
    return (
        r3[:n3].copy(),
        r4[:n4].copy(),
        off3,
        off4,
        g3[:n3].copy(),
        g4[:n4].copy(),
        c3f[:n3].copy(),
        c3b[:n3].copy(),
        c4f[:n4].copy(),
        c4b[:n4].copy(),
    )


@nb.njit(inline="always")
def _forward_sweep(
    I0,
    u,
    alphas_seg,
    kappas_seg,
    r3,
    r4,
    off3,
    off4,
    g3,
    g4,
    c3,
    c4,
    x_int,
    s_int,
    offsets,
    seg_bounds,
    Phi_f_b,
    Phi_b_b,
    t_j,
    rhoR_j,
    end,
    max_step,
):
    """
    One forward sweep over all segments (backward estimate ``u`` frozen).

    ``Phi_f_b[m, s]`` is the forward phase at the start of segment ``s``;
    ``Phi_b_b[m, q]`` the backward phase at the start of backward segment
    ``q`` (physical segment ``n_seg - 1 - q``), both of length ``n_seg + 1``.
    """
    n_modes = I0.shape[0]
    n_seg = seg_bounds.shape[0] - 1
    n_int = x_int.shape[0]
    A = np.empty((n_modes, n_int), dtype=np.complex128)
    y0 = I0.copy()
    for s in range(n_seg):
        q = n_seg - 1 - s
        a = offsets[s]
        b = offsets[s + 1]
        ca = n_int - offsets[s + 1]
        cb = n_int - offsets[s]
        if s > 0:
            # Junction j = s at x = seg_bounds[s]: A+_R = t A+_L + rho_R A-_R.
            for j in range(n_modes):
                A_ctr = u[j, cb - 1] * np.exp(1j * Phi_b_b[j, q + 1])
                y0[j] = t_j[s, j] * A[j, a - 1] + rhoR_j[s, j] * A_ctr * np.exp(
                    -1j * Phi_f_b[j, s]
                )
        phi0 = np.ascontiguousarray(Phi_f_b[:, s])
        ctr_phi0 = np.ascontiguousarray(Phi_b_b[:, q])
        seg_x = np.ascontiguousarray(x_int[a:b])
        sol = rk45_solve(
            bloch_fb_segment_rhs,
            (seg_x[0], seg_x[-1]),
            y0,
            args=(
                np.ascontiguousarray(alphas_seg[s]),
                np.ascontiguousarray(kappas_seg[s]),
                r3[off3[s] : off3[s + 1]],
                r4[off4[s] : off4[s + 1]],
                c3[off3[s] : off3[s + 1]],
                c4[off4[s] : off4[s + 1]],
                g3[off3[s] : off3[s + 1]],
                g4[off4[s] : off4[s + 1]],
                seg_x[0],
                phi0,
                np.ascontiguousarray(s_int[ca:cb]),
                np.ascontiguousarray(u[:, ca:cb]),
                s_int[ca],
                ctr_phi0,
                end,
            ),
            atol=SOLVER_ATOL,
            rtol=SOLVER_RTOL,
            max_num_steps=SOLVER_MAX_STEPS,
            t_eval=seg_x,
            first_step=SOLVER_FIRST_STEP,
            max_step=max_step,
        )
        A[:, a:b] = sol
    return A


@nb.njit(inline="always")
def _backward_sweep(
    I0,
    A_fwd,
    alphas_seg,
    kappas_seg,
    r3,
    r4,
    off3,
    off4,
    g3,
    g4,
    c3,
    c4,
    x_int,
    s_int,
    offsets,
    seg_bounds,
    Phi_f_b,
    Phi_b_b,
    tp_j,
    rhoL_j,
    end,
    max_step,
):
    """One backward sweep (mirrored coordinate) with the forward fields ``A_fwd``."""
    n_modes = I0.shape[0]
    n_seg = seg_bounds.shape[0] - 1
    n_int = x_int.shape[0]
    B = np.empty((n_modes, n_int), dtype=np.complex128)
    y0 = I0.copy()
    for q in range(n_seg):
        s = n_seg - 1 - q
        a = n_int - offsets[s + 1]
        b = n_int - offsets[s]
        fa = offsets[s]
        fb = offsets[s + 1]
        if q > 0:
            # Junction j = s + 1 at x = seg_bounds[s + 1], left segment s:
            # A-_L = t' A-_R + rho_L A+_L.
            jn = s + 1
            for j in range(n_modes):
                A_plus_L = A_fwd[j, fb - 1] * np.exp(1j * Phi_f_b[j, s + 1])
                y0[j] = tp_j[jn, j] * B[j, a - 1] + rhoL_j[jn, j] * A_plus_L * np.exp(
                    -1j * Phi_b_b[j, q]
                )
        phi0 = np.ascontiguousarray(Phi_b_b[:, q])
        ctr_phi0 = np.ascontiguousarray(Phi_f_b[:, s])
        seg_s = np.ascontiguousarray(s_int[a:b])
        sol = rk45_solve(
            bloch_fb_segment_rhs,
            (seg_s[0], seg_s[-1]),
            y0,
            args=(
                np.ascontiguousarray(alphas_seg[s]),
                np.ascontiguousarray(kappas_seg[s]),
                r3[off3[s] : off3[s + 1]],
                r4[off4[s] : off4[s + 1]],
                c3[off3[s] : off3[s + 1]],
                c4[off4[s] : off4[s + 1]],
                g3[off3[s] : off3[s + 1]],
                g4[off4[s] : off4[s + 1]],
                seg_s[0],
                phi0,
                np.ascontiguousarray(x_int[fa:fb]),
                np.ascontiguousarray(A_fwd[:, fa:fb]),
                x_int[fa],
                ctr_phi0,
                end,
            ),
            atol=SOLVER_ATOL,
            rtol=SOLVER_RTOL,
            max_num_steps=SOLVER_MAX_STEPS,
            t_eval=seg_s,
            first_step=SOLVER_FIRST_STEP,
            max_step=max_step,
        )
        B[:, a:b] = sol
    return B


@nb.njit(parallel=True, cache=True)
def _bloch_fb_solve_segmented(
    x_int,
    s_int,
    offsets,
    take,
    take_b,
    seg_bounds,
    y0_array_fwd,
    y0_array_bwd,
    gammas_seg,
    relations_3wm,
    relations_4wm,
    base_3wm,
    base_4wm,
    prefac_table,
    phi_plus_table,
    phi_minus_table,
    L_nl,
    positions,
    n_cells,
    t_table,
    tp_table,
    rhoL_table,
    rhoR_table,
    Zp_in_table,
    Zm_in_table,
    Zp_out_table,
    Zm_out_table,
    passes,
    kappa_cutoff,
    cell_phase_cutoff,
    m_max,
    convergence_threshold,
    save_all_passes,
    anderson_depth,
    pw3_table,
    pw4_table,
    Ze_table,
    Zout_in_table,
    Zout_out_table,
    apcz_in_table,
    apcz_out_table,
    tof_in_table,
    tof_out_table,
    max_step,
):
    """Forward/backward Bloch CME solve for all frequency points (segmented)."""
    n_freq = gammas_seg.shape[0]
    n_seg = gammas_seg.shape[1]
    n_modes = gammas_seg.shape[2]
    n_int = x_int.shape[0]
    n_x = take.shape[0]
    end = seg_bounds[n_seg]

    if save_all_passes:
        I_out = np.empty((2 * n_freq * passes, n_modes, n_x), dtype=np.complex128)
    else:
        I_out = np.empty((2 * n_freq, n_modes, n_x), dtype=np.complex128)
    converged_flags = np.zeros(n_freq, dtype=np.bool_)
    residuals = np.full(n_freq, np.inf)

    for i in nb.prange(n_freq):
        alphas_seg = np.ascontiguousarray(gammas_seg[i].real)  # (n_seg, n_modes)
        kappas_seg = np.ascontiguousarray(gammas_seg[i].imag)

        # Accumulated phases at segment starts (length n_seg + 1).
        Phi_f_b = np.zeros((n_modes, n_seg + 1), dtype=np.float64)
        Phi_b_b = np.zeros((n_modes, n_seg + 1), dtype=np.float64)
        for s in range(n_seg):
            ln = seg_bounds[s + 1] - seg_bounds[s]
            for j in range(n_modes):
                Phi_f_b[j, s + 1] = Phi_f_b[j, s] + kappas_seg[s, j] * ln
        for q in range(n_seg):
            s = n_seg - 1 - q
            ln = seg_bounds[s + 1] - seg_bounds[s]
            for j in range(n_modes):
                Phi_b_b[j, q + 1] = Phi_b_b[j, q] + kappas_seg[s, j] * ln
        phase_end = np.empty(n_modes, dtype=np.complex128)
        for j in range(n_modes):
            phase_end[j] = np.exp(1j * Phi_f_b[j, n_seg])

        # End boundaries (first / last segment Bloch impedances).
        Ze_i = Ze_table[i]
        Zout_in = Zout_in_table[i]
        Zout_out = Zout_out_table[i]
        Zp0 = Zp_in_table[i]
        Zm0 = Zm_in_table[i]
        ZpL = Zp_out_table[i]
        ZmL = Zm_out_table[i]
        t_in = 2.0 * Ze_i / (apcz_in_table[i] * (Zp0 + Zout_in))
        rho_in = (Zm0 - Zout_in) / (Zp0 + Zout_in)
        t_out = 2.0 * Ze_i / (apcz_out_table[i] * (ZmL + Zout_out))
        rho_out = (ZpL - Zout_out) / (ZmL + Zout_out)
        tof_in = tof_in_table[i]
        tof_out_signed = -tof_out_table[i]

        t_j = np.ascontiguousarray(t_table[i])
        tp_j = np.ascontiguousarray(tp_table[i])
        rhoL_j = np.ascontiguousarray(rhoL_table[i])
        rhoR_j = np.ascontiguousarray(rhoR_table[i])

        r3, r4, off3, off4, g3, g4, c3f, c3b, c4f, c4b = _segment_rows(
            kappas_seg,
            relations_3wm,
            relations_4wm,
            base_3wm,
            base_4wm,
            prefac_table[i],
            phi_plus_table[i],
            phi_minus_table[i],
            L_nl,
            positions,
            n_cells,
            pw3_table[i],
            pw4_table[i],
            kappa_cutoff,
            cell_phase_cutoff,
            m_max,
        )

        I_fwd = t_in * y0_array_fwd[i]
        I_bwd = np.zeros(n_modes, dtype=np.complex128)
        A_bwd_i = np.zeros((n_modes, n_int), dtype=np.complex128)

        converged = False
        stable_count = 0
        prev_fwd = np.zeros(n_modes, dtype=np.complex128)
        prev_bwd = np.zeros(n_modes, dtype=np.complex128)

        use_aa = anderson_depth > 0
        depth = anderson_depth if use_aa else 1
        dG = np.zeros((depth, n_modes, n_int), dtype=np.complex128)
        dF = np.zeros((depth, n_modes, n_int), dtype=np.complex128)
        g_last = np.zeros((n_modes, n_int), dtype=np.complex128)
        f_last = np.zeros((n_modes, n_int), dtype=np.complex128)
        n_hist = 0
        head = 0
        have_last = False
        res_last = np.inf
        scale_floor = 10.0 * SOLVER_ATOL

        A_fwd_x = np.empty((n_modes, n_x), dtype=np.complex128)
        A_bwd_x = np.empty((n_modes, n_x), dtype=np.complex128)

        u_k = A_bwd_i
        for k in range(passes):
            A_fwd_i = _forward_sweep(
                I_fwd,
                u_k,
                alphas_seg,
                kappas_seg,
                r3,
                r4,
                off3,
                off4,
                g3,
                g4,
                c3f,
                c4f,
                x_int,
                s_int,
                offsets,
                seg_bounds,
                Phi_f_b,
                Phi_b_b,
                t_j,
                rhoR_j,
                end,
                max_step,
            )

            for j in range(n_modes):
                I_bwd[j] = (
                    -t_out[j] * y0_array_bwd[i, j]
                    + rho_out[j] * A_fwd_i[j, n_int - 1] * phase_end[j]
                )

            A_bwd_i = _backward_sweep(
                I_bwd,
                A_fwd_i,
                alphas_seg,
                kappas_seg,
                r3,
                r4,
                off3,
                off4,
                g3,
                g4,
                c3b,
                c4b,
                x_int,
                s_int,
                offsets,
                seg_bounds,
                Phi_f_b,
                Phi_b_b,
                tp_j,
                rhoL_j,
                end,
                max_step,
            )

            # Map to the output grid (used by the plain convergence test and
            # for storage).
            for j in range(n_modes):
                for kx in range(n_x):
                    A_fwd_x[j, kx] = A_fwd_i[j, take[kx]]
                    A_bwd_x[j, kx] = A_bwd_i[j, take_b[kx]]

            f_k = A_bwd_i - u_k
            scales = _mode_scales(A_bwd_i, scale_floor) + _mode_scales(A_fwd_i, 0.0)
            res = _relative_residual(f_k, scales)
            residuals[i] = res
            if use_aa:
                if not np.isfinite(res):
                    converged = False
                elif k > 0 and res < convergence_threshold:
                    converged = True
                if not converged and np.isfinite(res):
                    if res > 10.0 * res_last:
                        n_hist = 0
                        have_last = False
                    if have_last:
                        dG[head] = A_bwd_i - g_last
                        dF[head] = f_k - f_last
                        head = (head + 1) % depth
                        if n_hist < depth:
                            n_hist += 1
                    g_last[:, :] = A_bwd_i
                    f_last[:, :] = f_k
                    have_last = True
                    res_last = res
                    if n_hist > 0:
                        u_next = _anderson_mix(
                            A_bwd_i, f_k, dG, dF, n_hist, head, depth, scales
                        )
                    else:
                        u_next = A_bwd_i.copy()
                else:
                    u_next = A_bwd_i.copy()
            else:
                u_next = A_bwd_i
                if k > 0:
                    stable_count, converged = _convergence_step(
                        A_fwd_x,
                        A_bwd_x,
                        prev_fwd,
                        prev_bwd,
                        n_modes,
                        convergence_threshold,
                        stable_count,
                    )

            for j in range(n_modes):
                I_fwd[j] = (
                    t_in[j] * y0_array_fwd[i, j]
                    + rho_in[j] * u_next[j, n_int - 1] * phase_end[j]
                )

            id_add_fwd = 0
            id_add_bwd = n_freq
            if save_all_passes:
                id_add_fwd = k * n_freq
                id_add_bwd = k * n_freq + n_freq * passes
            _store_pass(
                I_out,
                A_fwd_x,
                A_bwd_x,
                rho_out,
                rho_in,
                tof_out_signed,
                tof_in,
                i + id_add_fwd,
                i + id_add_bwd,
                n_modes,
            )

            for j in range(n_modes):
                prev_fwd[j] = A_fwd_x[j, n_x - 1]
                prev_bwd[j] = A_bwd_x[j, n_x - 1]
            u_k = np.ascontiguousarray(u_next)

            if converged:
                converged_flags[i] = True
                if save_all_passes:
                    for kk in range(k + 1, passes):
                        _store_pass(
                            I_out,
                            A_fwd_x,
                            A_bwd_x,
                            rho_out,
                            rho_in,
                            tof_out_signed,
                            tof_in,
                            i + kk * n_freq,
                            i + kk * n_freq + n_freq * passes,
                            n_modes,
                        )
                break

    return I_out, converged_flags, residuals


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #


def bloch_fb_segmented_cmes_solve(
    x_array: FloatArray,
    seg_bounds: FloatArray,
    y0_array_fwd: ComplexArray,
    y0_array_bwd: ComplexArray,
    gammas_seg: ComplexArray,
    relations_3wm: np.ndarray,
    relations_4wm: np.ndarray,
    base_3wm: ComplexArray,
    base_4wm: ComplexArray,
    prefac: ComplexArray,
    phi_plus: ComplexArray,
    phi_minus: ComplexArray,
    L_nl: FloatArray,
    positions: FloatArray,
    n_cells: np.ndarray,
    Zb_plus: ComplexArray,
    Zb_minus: ComplexArray,
    passes: int = 10,
    kappa_cutoff: float = 0.2,
    cell_phase_cutoff: float | None = None,
    m_max: int = 5,
    convergence_threshold: float = 0.01,
    save_all_passes: bool = False,
    anderson_depth: int = 5,
    process_weights_3wm: FloatArray | None = None,
    process_weights_4wm: FloatArray | None = None,
    Ze: float | ComplexArray = 50.0,
    Zout_in: ComplexArray | None = None,
    Zout_out: ComplexArray | None = None,
    abcd_in: ComplexArray | None = None,
    abcd_out: ComplexArray | None = None,
    max_step: float = np.inf,
) -> ComplexArray:
    """
    Solve the forward/backward Bloch CMEs of a segmented device.

    Parameters
    ----------
    x_array : (n_x,) float
        Output grid on ``[0, L]``; must be symmetric under ``x -> L - x``
        (e.g. ``linspace(0, L, n)``), as for the uniform kernel.
    seg_bounds : (n_seg + 1,) float
        Segment boundaries in unit cells, ``seg_bounds[-1] = L``. Every
        segment must hold a whole number of its own supercells.
    gammas_seg : (n_freq, n_seg, n_modes) complex
        Kernel propagation constants (``-alpha - j k``) of every segment.
    base_3wm, base_4wm : (n_seg, n_rel)
        RWA multiplicity/4 times the segment's epsilon / xi.
    prefac : (n_freq, n_seg, n_modes)
        ``-2j omega / W`` of every segment.
    phi_plus, phi_minus : (n_freq, n_seg, n_modes, n_nl)
        Intracell current profiles (padded with zeros to the largest ``n_nl``).
    L_nl, positions : (n_seg, n_nl)
        Nonlinear inductances (0 for padding) and positions inside the
        segment's supercell (unit cells).
    n_cells : (n_seg,) int
        Unit cells per supercell of every segment.
    Zb_plus, Zb_minus : (n_freq, n_seg, n_modes)
        Bloch impedances of every segment; consecutive segments define the
        junction coefficients (:func:`junction_coefficients`), the first and
        last ones the end boundaries.
    process_weights_3wm, process_weights_4wm : (n_freq, n_seg, n_rel), optional
        Stopband-edge weights per segment; default 1.

    Other parameters as in :func:`twpasolver.cmes.bloch_fb.bloch_fb_cmes_solve`.

    Returns
    -------
    ndarray, shape (2 * n_freq [* passes], n_modes, n_x)
        Same layout as the uniform solver: slow envelopes on ``x_array``
        (forward on ``x``, backward on the mirrored coordinate), last point
        replaced by the current coupled out to the external line. Restore the
        propagation phase with :func:`segmented_phase`.
    """
    if cell_phase_cutoff is None:
        cell_phase_cutoff = solver_config.BLOCH_CELL_PHASE_CUTOFF
    gammas_seg = np.asarray(gammas_seg, dtype=np.complex128)
    if gammas_seg.ndim != 3:
        raise ValueError("gammas_seg must have shape (n_freq, n_seg, n_modes).")
    n_freq, n_seg, n_modes = gammas_seg.shape
    seg_bounds = np.asarray(seg_bounds, dtype=np.float64)
    x_array = np.asarray(x_array, dtype=np.float64)
    if len(seg_bounds) != n_seg + 1:
        raise ValueError("seg_bounds must have n_seg + 1 entries.")
    if not np.isclose(x_array[0], 0.0) or not np.isclose(x_array[-1], seg_bounds[-1]):
        raise ValueError("x_array must span [0, seg_bounds[-1]].")
    if not np.allclose(x_array, seg_bounds[-1] - x_array[::-1]):
        raise ValueError("x_array must be symmetric under x -> L - x.")
    n_cells = np.asarray(n_cells, dtype=np.int64)
    lengths = np.diff(seg_bounds)
    if np.any(np.abs(lengths / n_cells - np.round(lengths / n_cells)) > 1e-9):
        raise ValueError("Every segment must hold a whole number of its supercells.")

    x_int, offsets, take = build_segment_grid(x_array, seg_bounds)
    s_int = seg_bounds[-1] - x_int[::-1]
    # Backward envelopes live on the mirrored internal grid, which is the
    # internal grid of the mirrored device (x_array is symmetric).
    s_int_chk, _, take_b = build_segment_grid(
        x_array, seg_bounds[-1] - seg_bounds[::-1]
    )
    if not np.allclose(s_int_chk, s_int):
        raise ValueError("Internal grid construction is not mirror-consistent.")

    Zp = np.asarray(Zb_plus, dtype=np.complex128)
    Zm = np.asarray(Zb_minus, dtype=np.complex128)
    t, tp, rho_L, rho_R = junction_coefficients(Zp, Zm)

    def _table(val, default):
        if val is None:
            return default
        if np.ndim(val) == 0:
            return np.full((n_freq, n_modes), complex(val), dtype=np.complex128)
        arr = np.asarray(val, dtype=np.complex128)
        if arr.shape != (n_freq, n_modes):
            raise ValueError(
                f"Boundary table must have shape {(n_freq, n_modes)}, got {arr.shape}."
            )
        return arr

    Ze_table = _table(Ze, None)
    Zout_in_table = _table(Zout_in, Ze_table)
    Zout_out_table = _table(Zout_out, Ze_table)

    def _apcz_tof(abcd):
        if abcd is None:
            ones = np.ones((n_freq, n_modes), dtype=np.complex128)
            return ones, ones.copy()
        tab = np.asarray(abcd, dtype=np.complex128)
        A, B, C, D = (tab[..., k] for k in range(4))
        apcz = A + C * Ze_table
        return apcz, (A * D - B * C) / apcz

    apcz_in, tof_in = _apcz_tof(abcd_in)
    apcz_out, tof_out = _apcz_tof(abcd_out)

    def _weights(val, n_rel):
        if val is None:
            return np.ones((n_freq, n_seg, n_rel), dtype=np.float64)
        arr = np.asarray(val, dtype=np.float64)
        if arr.shape != (n_freq, n_seg, n_rel):
            raise ValueError(
                f"Process weights must have shape {(n_freq, n_seg, n_rel)}, "
                f"got {arr.shape}."
            )
        return arr

    pw3 = _weights(process_weights_3wm, len(relations_3wm))
    pw4 = _weights(process_weights_4wm, len(relations_4wm))

    c = np.ascontiguousarray
    I_out, converged, residuals = _bloch_fb_solve_segmented(
        c(x_int),
        c(s_int),
        c(offsets),
        c(take),
        c(take_b),
        c(seg_bounds),
        c(y0_array_fwd, dtype=np.complex128),
        c(y0_array_bwd, dtype=np.complex128),
        c(gammas_seg),
        c(relations_3wm, dtype=np.int64),
        c(relations_4wm, dtype=np.int64),
        c(base_3wm, dtype=np.complex128),
        c(base_4wm, dtype=np.complex128),
        c(prefac, dtype=np.complex128),
        c(phi_plus, dtype=np.complex128),
        c(phi_minus, dtype=np.complex128),
        c(L_nl, dtype=np.float64),
        c(positions, dtype=np.float64),
        c(n_cells),
        c(t),
        c(tp),
        c(rho_L),
        c(rho_R),
        c(Zp[:, 0, :]),
        c(Zm[:, 0, :]),
        c(Zp[:, -1, :]),
        c(Zm[:, -1, :]),
        int(passes),
        float(kappa_cutoff),
        float(cell_phase_cutoff),
        int(m_max),
        float(convergence_threshold),
        bool(save_all_passes),
        int(anderson_depth),
        c(pw3),
        c(pw4),
        c(Ze_table),
        c(Zout_in_table),
        c(Zout_out_table),
        c(apcz_in),
        c(apcz_out),
        c(tof_in),
        c(tof_out),
        float(max_step),
    )
    if not np.all(converged):
        bad = np.flatnonzero(~converged)
        log.warning(
            "Segmented Bloch forward-backward CMEs did not converge for %d of %d "
            "frequency points within %d passes (indices %s; last relative "
            "residuals %s).",
            bad.size,
            n_freq,
            passes,
            bad[:20].tolist(),
            np.array2string(
                residuals[bad[:20]], formatter={"float_kind": lambda v: f"{v:.1e}"}
            ),
        )
    return I_out
