"""
Forward-backward coupled mode equations in the Bloch-mode basis.

Extension of :mod:`twpasolver.cmes.bloch` to counter-propagating waves and
reflections at the device terminations (uniform devices only).

Physics summary (package convention exp(+j omega t), forward ~ exp(-gamma x)):

* Every mode is expanded exactly as ``psi_n = A+_n psi+_n + A-_n psi-_n``,
  with ``psi+_n(0) = (Zb+_n, 1)`` and ``psi-_n(0) = (-Zb-_n, 1)``. Both
  amplitudes are currents referenced to +x, so the total current through a
  nonlinear inductor is simply ``A+ phi+ e^{-gamma x} + A- phi- e^{+gamma x}``.
* Projecting the nonlinear series source with the reciprocity invariant
  ``W_n = <psi+_n, psi-_n> = Zb+_n + Zb-_n`` gives
  ``dA+_n/dx = -(j omega_n / W_n) <L Pi_n phi-_n> e^{+gamma x}`` and
  ``dA-_n/dx = +(j omega_n / W_n) <L Pi_n phi+_n> e^{-gamma x}``.
  The backward solve runs in the mirrored coordinate s = L - x, which flips
  the sign of the second equation: both solves use the prefactor
  ``-2j omega_n / W_n`` (times the RWA multiplicity/4 and epsilon or xi),
  and differ only in the partner profile (phi- for the forward target, phi+
  for the backward one) and in which profile each input carries
  (own direction vs counter direction).
* Because the expansion fixes voltage and current at every supercell
  boundary, ``V = Zb+ A+ - Zb- A-`` and ``I = A+ + A-`` hold exactly at the
  device ends. The boundary coefficients are therefore constant (no
  derivative correction, no damping of a self-consistent update):

    input  (x=0):  A+ = t_in a_in + rho_in A-,
                   t_in = 2 Ze / (apcz_in (Zb+ + Zout_in)),
                   rho_in = (Zb- - Zout_in) / (Zb+ + Zout_in),
                   b_in = tof_in (1 + rho_in) A-
    output (x=L):  A- = -t_out a_out + rho_out A+,
                   t_out = 2 Ze / (apcz_out (Zb- + Zout_out)),
                   rho_out = (Zb+ - Zout_out) / (Zb- + Zout_out),
                   b_out = -tof_out (1 + rho_out) A+

  where ``a``/``b`` are the incident/outgoing currents in the external line,
  ``apcz = A + C Ze``, ``Zout = (B + D Ze)/(A + C Ze)`` and
  ``tof = det/apcz`` of each connector (port 2 facing the device). The minus
  signs at the output end come from referencing the backward current to +x.

The slow variables are the same as in the standard kernels: only the
propagation phase exp(i kappa x) is factored out, the loss stays in the
envelope through the ``alpha * A`` term.

Iteration: each pass solves the forward envelopes with the backward ones of
the previous pass, then the backward envelopes. The pass is a map u -> G(u)
on the backward envelopes u. Strong counter-propagating coupling (e.g. a
Bragg-matched forward/backward pair near a stopband) puts eigenvalues of this
map outside the unit circle and the plain iteration diverges even though the
boundary-value problem has a regular solution. Anderson acceleration
(``anderson_depth`` previous passes, per-mode weighted least squares) solves
for the fixed point instead and converges in that regime as well. It cannot
cure an instability inside a single sweep (a mode close to a stopband edge
whose forward-only subsystem grows exponentially).
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
from twpasolver.cmes.base import _sorted_interp_bracket
from twpasolver.cmes.cmes_fb import _convergence_step, _store_pass
from twpasolver.cmes.rk45 import rk45_solve
from twpasolver.cmes.solver_config import (
    SOLVER_ATOL,
    SOLVER_FIRST_STEP,
    SOLVER_MAX_STEPS,
    SOLVER_RTOL,
)
from twpasolver.logger import log


@nb.njit(cache=True)
def _cubic_stencil(xp, x):
    """
    First index and weights of the 4-point Lagrange stencil around ``x``.

    The stored envelopes oscillate with the residual mismatch of the kept
    terms, up to ``kappa_cutoff * pi`` per supercell; 4-point cubic
    interpolation keeps the error of a table lookup at about ``phi^4 / 50``
    for a phase ``phi`` per grid step (linear: ``phi^2 / 8``) without storing
    anything more. Outside the grid the value is clamped as in ``np.interp``;
    grids shorter than 4 points fall back to linear interpolation. Unused
    weights are zero.
    """
    n = len(xp)
    lo, hi = _sorted_interp_bracket(xp, x)
    if lo == hi:
        return lo, 1.0, 0.0, 0.0, 0.0
    if n < 4:
        frac = (x - xp[lo]) / (xp[hi] - xp[lo])
        return lo, 1.0 - frac, frac, 0.0, 0.0
    i0 = min(max(lo - 1, 0), n - 4)
    x0, x1, x2, x3 = xp[i0], xp[i0 + 1], xp[i0 + 2], xp[i0 + 3]
    d0, d1, d2, d3 = x - x0, x - x1, x - x2, x - x3
    w0 = d1 * d2 * d3 / ((x0 - x1) * (x0 - x2) * (x0 - x3))
    w1 = d0 * d2 * d3 / ((x1 - x0) * (x1 - x2) * (x1 - x3))
    w2 = d0 * d1 * d3 / ((x2 - x0) * (x2 - x1) * (x2 - x3))
    w3 = d0 * d1 * d2 / ((x3 - x0) * (x3 - x1) * (x3 - x2))
    return i0, w0, w1, w2, w3


@nb.njit(cache=True)
def _stencil_value(arr, r, i0, w0, w1, w2, w3):
    """Row ``r`` of ``arr`` (rows, n_x) at the point of :func:`_cubic_stencil`."""
    v = w0 * arr[r, i0]
    if w1 != 0.0:
        v += w1 * arr[r, i0 + 1]
    if w2 != 0.0:
        v += w2 * arr[r, i0 + 2]
    if w3 != 0.0:
        v += w3 * arr[r, i0 + 3]
    return v


@nb.njit(cache=True)
def _filter_wm_terms_bloch(
    kappas, relations_3wm, relations_4wm, cutoff, n_cells, m_lim
):
    """
    Keep the phase-matched (sign variant, reciprocal vector) pairs of every relation.

    Each sign variant of a relation has a mismatch ``dk``; each Fourier
    component ``G_m = 2 pi m / d`` (``d = n_cells``) of the intracell coupling
    profile turns it into a term with residual mismatch ``dk + G_m``. A pair is
    kept when

        |dk + G_m| < cutoff * k_ref,    k_ref = min(k_proc, pi / d),

    with ``k_proc`` the largest ``|kappa|`` among the modes of the process: the
    mismatch is measured relative to the process wavenumber, capped at the zone
    edge, the largest wavenumber the supercell-averaged envelopes resolve. The
    threshold never exceeds ``pi / d``, so every term is assigned to a single
    ``G_m`` (the nearest one).
    ``m_lim`` (n_modes,) is the harmonic limit of every mode; a row uses
    ``|m| <= max(m_lim)`` over its modes. Partner terms of a process contain
    the same modes and have opposite ``dk``, so they pair with opposite ``m``
    within the same symmetric range and the same threshold, and are kept or
    dropped together. ``m_lim = 0`` gives the cell-averaged model.

    Returns rows ``[target, field indices..., base relation index, m]``.
    """
    num_modes = len(kappas)
    n_g = 2 * np.max(m_lim) + 1 if num_modes > 0 else 1
    g_unit = 2.0 * np.pi / n_cells
    k_edge = np.pi / n_cells
    out3 = np.empty((len(relations_3wm) * 4 * n_g, 5), dtype=relations_3wm.dtype)
    out4 = np.empty((len(relations_4wm) * 8 * n_g, 6), dtype=relations_4wm.dtype)
    n3 = 0
    n4 = 0
    for r in range(len(relations_3wm)):
        rel = relations_3wm[r]
        k_proc = abs(kappas[rel[0]])
        m_row = m_lim[rel[0]]
        for q in range(2):
            m = rel[1 + q]
            base = m - num_modes if m >= num_modes else m
            k_proc = max(k_proc, abs(kappas[base]))
            m_row = max(m_row, m_lim[base])
        thr = min(cutoff * min(k_proc, k_edge), k_edge)
        for v in range(4):
            dk = -kappas[rel[0]]
            idx = np.empty(2, dtype=relations_3wm.dtype)
            for q in range(2):
                counter = (v >> q) & 1
                m = rel[1 + q]
                conj = m >= num_modes
                base = m - num_modes if conj else m
                sign = -1.0 if conj else 1.0
                if counter:
                    sign = -sign
                dk += sign * kappas[base]
                idx[q] = m + (2 * num_modes if counter else 0)
            for mg in range(-m_row, m_row + 1):
                res = abs(dk + mg * g_unit)
                if res < thr:
                    out3[n3, 0] = rel[0]
                    out3[n3, 1] = idx[0]
                    out3[n3, 2] = idx[1]
                    out3[n3, 3] = r
                    out3[n3, 4] = mg
                    n3 += 1
    for r in range(len(relations_4wm)):
        rel = relations_4wm[r]
        k_proc = abs(kappas[rel[0]])
        m_row = m_lim[rel[0]]
        for q in range(3):
            m = rel[1 + q]
            base = m - num_modes if m >= num_modes else m
            k_proc = max(k_proc, abs(kappas[base]))
            m_row = max(m_row, m_lim[base])
        thr = min(cutoff * min(k_proc, k_edge), k_edge)
        for v in range(8):
            dk = -kappas[rel[0]]
            idx = np.empty(3, dtype=relations_4wm.dtype)
            for q in range(3):
                counter = (v >> q) & 1
                m = rel[1 + q]
                conj = m >= num_modes
                base = m - num_modes if conj else m
                sign = -1.0 if conj else 1.0
                if counter:
                    sign = -sign
                dk += sign * kappas[base]
                idx[q] = m + (2 * num_modes if counter else 0)
            for mg in range(-m_row, m_row + 1):
                res = abs(dk + mg * g_unit)
                if res < thr:
                    out4[n4, 0] = rel[0]
                    out4[n4, 1] = idx[0]
                    out4[n4, 2] = idx[1]
                    out4[n4, 3] = idx[2]
                    out4[n4, 4] = r
                    out4[n4, 5] = mg
                    n4 += 1
    return out3[:n3], out4[:n4]


@nb.njit(cache=True)
def _bloch_row_coeffs(
    rows,
    base_coeffs,
    prefac,
    phi_plus,
    phi_minus,
    L_nl,
    positions,
    n_cells,
    own_forward,
):
    """
    Coefficient of every kept (sign-expanded) mixing row for one solve direction.

    Parameters
    ----------
    rows : (n_rows, n_in + 3) int
        Rows from ``_filter_wm_terms_bloch``: target mode, ``n_in`` indices into the
        field vector of the RHS (blocks of ``n_modes``: own, conj own, counter,
        conj counter), base relation index, reciprocal-vector index ``m``.
    base_coeffs : (n_rel,) complex
        RWA multiplicity/4 times epsilon (3WM) or xi (4WM).
    prefac : (n_modes,) complex
        ``-2j omega_n / W_n``.
    phi_plus, phi_minus : (n_modes, n_nl) complex
        Periodic current profiles through the nonlinear inductors.
    L_nl : (n_nl,) float
    positions : (n_nl,) float
        Position of each nonlinear inductor in the supercell (unit cells).
    n_cells : int
        Unit cells per supercell.
    own_forward : bool
        True for the forward solve, False for the backward one.

    The coefficient of a row with index ``m`` is the Fourier component
    ``(1/d) sum_j F_j exp(-i G x_j)`` of its intracell profile, with
    ``G = 2 pi m / d`` in the solve coordinate. The backward solve runs in the
    mirrored coordinate ``s = L - x``, where ``exp(i G s)`` corresponds to the
    physical component ``-G``, hence the sign flip of the phase below.
    """
    n_rows = rows.shape[0]
    n_in = rows.shape[1] - 3
    g_unit = 2.0 * np.pi / n_cells
    phase_sign = -1.0 if own_forward else 1.0
    n_modes = phi_plus.shape[0]
    n_nl = L_nl.shape[0]
    out = np.zeros(n_rows, dtype=np.complex128)
    for r in range(n_rows):
        target = rows[r, 0]
        base = base_coeffs[rows[r, n_in + 1]]
        if base == 0:
            continue
        g = rows[r, n_in + 2] * g_unit
        acc = 0j
        for j in range(n_nl):
            if own_forward:
                v = L_nl[j] * phi_minus[target, j]
            else:
                v = L_nl[j] * phi_plus[target, j]
            for q in range(n_in):
                idx = rows[r, 1 + q]
                block = idx // n_modes
                m = idx - block * n_modes
                same_direction = block < 2
                forward = same_direction if own_forward else not same_direction
                val = phi_plus[m, j] if forward else phi_minus[m, j]
                if block == 1 or block == 3:
                    val = np.conj(val)
                v = v * val
            acc += v * np.exp(1j * phase_sign * g * positions[j])
        out[r] = base * prefac[target] * acc / n_cells
    return out


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
        nb_float1d,
        nb_complex2d,
        nb.float64,
    ),
    cache=True,
)
def bloch_fb_rhs(
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
    x_evals,
    currents_evals,
    end,
):
    """
    RHS of the Bloch forward/backward CMEs for one solve direction.

    Same field bookkeeping as ``cme_rhs`` (own and counter-propagating
    carriers, counter field read at ``end - x``); coefficients are given per
    kept row and already include the per-mode prefactor. ``row_g_*`` is the
    reciprocal vector of each row; its factor ``exp(i G x)`` makes the row
    slowly varying (``|dk + G|`` small, enforced by the filter).
    """
    n = len(currents)
    out = np.zeros(n, nb.complex128)
    ep = np.exp(1j * kappas * x)
    en = np.exp(1j * kappas * (end - x))

    ar = np.empty(4 * n, dtype=nb.complex128)
    x_ctr = end - x
    i0, w0, w1, w2, w3 = _cubic_stencil(x_evals, x_ctr)
    for i in range(n):
        ctr = _stencil_value(currents_evals, i, i0, w0, w1, w2, w3) * en[i]
        ar[i] = currents[i] * ep[i]
        ar[i + n] = np.conj(ar[i])
        ar[i + 2 * n] = ctr
        ar[i + 3 * n] = np.conj(ctr)

    for r in range(rows_3wm.shape[0]):
        term = row_coeffs_3wm[r] * ar[rows_3wm[r, 1]] * ar[rows_3wm[r, 2]]
        if row_g_3wm[r] != 0.0:
            term *= np.exp(1j * row_g_3wm[r] * x)
        out[rows_3wm[r, 0]] += term
    for r in range(rows_4wm.shape[0]):
        term = (
            row_coeffs_4wm[r]
            * ar[rows_4wm[r, 1]]
            * ar[rows_4wm[r, 2]]
            * ar[rows_4wm[r, 3]]
        )
        if row_g_4wm[r] != 0.0:
            term *= np.exp(1j * row_g_4wm[r] * x)
        out[rows_4wm[r, 0]] += term

    for i in range(n):
        out[i] = out[i] / ep[i] + alphas[i] * currents[i]
    return out


# --------------------------------------------------------------------------- #
# Edge-basis modes: predicted in the sweeps, solved exactly once per pass
# --------------------------------------------------------------------------- #
#
# A mode flagged as close to a stopband edge is expanded on a fixed Bloch pair
# (see twpasolver.cmes.bloch.edge_basis_tables). Its amplitudes c = (c+, c-),
# in the physical coordinate and with the edge phase exp(-+j kappa_e x)
# factored out, obey dc/dx = K c + S(x), with K constant and S the nonlinear
# source. The Bragg term K[0,1], K[1,0] couples the two directions strongly
# (|K01| L up to ~100): split between a forward and a backward sweep it is a
# Neumann series that diverges. Each pass therefore
#   1. forward sweep: c+ is integrated together with the other modes, with
#      its linear terms K00 c+ + K01 c-(x) and c- from the current iterate
#      (predictor: the forward conversion processes stay inside the sweep);
#   2. backward sweep: the flagged mode is frozen at the current iterate;
#   3. the source S of the flagged mode is evaluated from the swept fields,
#      and the 2x2 two-point problem is solved exactly (discrete Riccati
#      sweep); the result replaces both envelopes (corrector).
# The iterate seen by Anderson acceleration is unchanged (all backward
# envelopes). Storage of the backward envelope is the usual mirrored one:
#   c-(x) = B(L - x) * exp(j kappas L)   (kappas = -kappa_e in the kernel).


@nb.njit(cache=True)
def bloch_fb_rhs_edge(
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
    x_evals,
    currents_evals,
    end,
    edge_code,
    own_evals,
    lin_own,
    lin_ctr,
    ctr_phase,
):
    """
    :func:`bloch_fb_rhs` with the edge-basis modes of one solve direction.

    ``edge_code[i]``: 0 ordinary mode; 1 frozen (own field read from
    ``own_evals`` at ``x``, zero derivative); 2 integrated with the extra
    linear terms ``lin_own[i] * c + lin_ctr[i] * ctr_phase[i] * u_i(end - x)``,
    with ``u_i`` the stored counter envelope (``currents_evals``), so that
    ``ctr_phase * u(end - x)`` is the counter amplitude in the solve
    coordinate.
    """
    n = len(currents)
    cur = currents.copy()
    any_frozen = False
    any_lin = False
    for i in range(n):
        if edge_code[i] == 1.0:
            any_frozen = True
        elif edge_code[i] == 2.0:
            any_lin = True
    if any_frozen:
        i0, w0, w1, w2, w3 = _cubic_stencil(x_evals, x)
        for i in range(n):
            if edge_code[i] == 1.0:
                cur[i] = _stencil_value(own_evals, i, i0, w0, w1, w2, w3)
    out = bloch_fb_rhs(
        x,
        cur,
        alphas,
        kappas,
        rows_3wm,
        rows_4wm,
        row_coeffs_3wm,
        row_coeffs_4wm,
        row_g_3wm,
        row_g_4wm,
        x_evals,
        currents_evals,
        end,
    )
    j0, v0, v1, v2, v3 = 0, 0.0, 0.0, 0.0, 0.0
    if any_lin:
        j0, v0, v1, v2, v3 = _cubic_stencil(x_evals, end - x)
    for i in range(n):
        if edge_code[i] == 1.0:
            out[i] = 0.0
        elif edge_code[i] == 2.0:
            ctr = _stencil_value(currents_evals, i, j0, v0, v1, v2, v3)
            out[i] += lin_own[i] * cur[i] + lin_ctr[i] * ctr_phase[i] * ctr
    return out


@nb.njit(cache=True)
def _expm_2x2(K):
    """``exp(K)`` of a 2x2 matrix in closed form, regular when ``K - tr(K)/2`` is nilpotent."""
    tau = 0.5 * (K[0, 0] + K[1, 1])
    n00 = K[0, 0] - tau
    n01 = K[0, 1]
    n10 = K[1, 0]
    mu2 = n00 * n00 + n01 * n10  # (K - tau I)^2 = mu2 I
    if np.abs(mu2) < 1e-8:
        ch = 1.0 + mu2 / 2.0 + mu2 * mu2 / 24.0
        sh = 1.0 + mu2 / 6.0 + mu2 * mu2 / 120.0
    else:
        mu = np.sqrt(mu2 + 0j)
        ch = np.cosh(mu)
        sh = np.sinh(mu) / mu
    e = np.exp(tau)
    out = np.empty((2, 2), dtype=np.complex128)
    out[0, 0] = e * (ch + sh * n00)
    out[0, 1] = e * sh * n01
    out[1, 0] = e * sh * n10
    out[1, 1] = e * (ch - sh * n00)
    return out


@nb.njit(cache=True)
def _edge_bvp_solve(K, h, S_plus, S_minus, cp_in, rho_in, cm_out, rho_out):
    """
    Exact solution of ``dc/dx = K c + S`` on a uniform grid with mixed end conditions.

    ``c+(0) = cp_in + rho_in c-(0)`` and ``c-(L) = cm_out + rho_out c+(L)``.
    The homogeneous step is exact (``exp(K h)``); the source is integrated
    with the trapezoidal rule. The two-point problem is solved by the discrete
    Riccati (invariant-imbedding) sweep ``c-_j = R_j c+_j + Q_j``: ``R`` is the
    reflection coefficient of the line to the right of ``x_j``, built from the
    output end, so the sweep is stable inside a stopband as well.

    Returns ``(c_plus, c_minus)`` on the grid.
    """
    n = S_plus.shape[0]
    P = _expm_2x2(K * h)
    a, b, c, d = P[0, 0], P[0, 1], P[1, 0], P[1, 1]
    sp = np.empty(n, dtype=np.complex128)
    sm = np.empty(n, dtype=np.complex128)
    for j in range(n - 1):
        sp[j] = 0.5 * h * (a * S_plus[j] + b * S_minus[j] + S_plus[j + 1])
        sm[j] = 0.5 * h * (c * S_plus[j] + d * S_minus[j] + S_minus[j + 1])
    R = np.empty(n, dtype=np.complex128)
    Q = np.empty(n, dtype=np.complex128)
    R[n - 1] = rho_out
    Q[n - 1] = cm_out
    for j in range(n - 2, -1, -1):
        den = d - R[j + 1] * b
        R[j] = (R[j + 1] * a - c) / den
        Q[j] = (R[j + 1] * sp[j] + Q[j + 1] - sm[j]) / den
    c_plus = np.empty(n, dtype=np.complex128)
    c_minus = np.empty(n, dtype=np.complex128)
    c_plus[0] = (cp_in + rho_in * Q[0]) / (1.0 - rho_in * R[0])
    c_minus[0] = R[0] * c_plus[0] + Q[0]
    for j in range(n - 1):
        c_plus[j + 1] = a * c_plus[j] + b * c_minus[j] + sp[j]
        c_minus[j + 1] = R[j + 1] * c_plus[j + 1] + Q[j + 1]
    return c_plus, c_minus


@nb.njit(cache=True)
def _interp_rows(arr, x_evals, x):
    """Cubic interpolation of every row of ``arr`` (rows, n_x) at ``x``."""
    i0, w0, w1, w2, w3 = _cubic_stencil(x_evals, x)
    out = np.empty(arr.shape[0], dtype=np.complex128)
    for r in range(arr.shape[0]):
        out[r] = _stencil_value(arr, r, i0, w0, w1, w2, w3)
    return out


@nb.njit(cache=True)
def _rows_with_targets(rows, coeffs, gs, is_target):
    """Subset of kept rows whose target mode has ``is_target > 0``."""
    keep = 0
    for r in range(rows.shape[0]):
        if is_target[rows[r, 0]] > 0.0:
            keep += 1
    rows_out = np.empty((keep, rows.shape[1]), dtype=rows.dtype)
    c_out = np.empty(keep, dtype=np.complex128)
    g_out = np.empty(keep, dtype=np.float64)
    q = 0
    for r in range(rows.shape[0]):
        if is_target[rows[r, 0]] > 0.0:
            rows_out[q] = rows[r]
            c_out[q] = coeffs[r]
            g_out[q] = gs[r]
            q += 1
    return rows_out, c_out, g_out


@nb.njit(cache=True)
def _edge_sources(
    xs,
    flag_idx,
    is_flag,
    kappas,
    r3,
    r4,
    c3_fwd,
    c4_fwd,
    c3_bwd,
    c4_bwd,
    g3,
    g4,
    x_evals,
    A_fwd,
    A_bwd,
    x_end,
    phase_end,
):
    """
    Nonlinear sources ``S+-(x)`` of the flagged modes, physical coordinate.

    ``S+`` is the forward-solve derivative of ``c+``; ``S-`` follows from the
    backward-solve derivative ``dB/ds`` at ``s = L - x`` through
    ``c-(x) = B(L - x) exp(j kappas L)``: ``S-(x) = -exp(j kappas L) dB/ds``.
    Only rows targeting a flagged mode are evaluated.
    """
    n_modes = A_fwd.shape[0]
    n_flag = flag_idx.shape[0]
    n = xs.shape[0]
    zeros_alpha = np.zeros(n_modes, dtype=np.float64)
    r3f, c3f, g3f = _rows_with_targets(r3, c3_fwd, g3, is_flag)
    r4f, c4f, g4f = _rows_with_targets(r4, c4_fwd, g4, is_flag)
    _, c3b, _ = _rows_with_targets(r3, c3_bwd, g3, is_flag)
    _, c4b, _ = _rows_with_targets(r4, c4_bwd, g4, is_flag)
    S_plus = np.zeros((n_flag, n), dtype=np.complex128)
    S_minus = np.zeros((n_flag, n), dtype=np.complex128)
    for j in range(n):
        x = xs[j]
        cur = _interp_rows(A_fwd, x_evals, x)
        out = bloch_fb_rhs(
            x,
            cur,
            zeros_alpha,
            kappas,
            r3f,
            r4f,
            c3f,
            c4f,
            g3f,
            g4f,
            x_evals,
            A_bwd,
            x_end,
        )
        s = x_end - x
        cur_b = _interp_rows(A_bwd, x_evals, s)
        out_b = bloch_fb_rhs(
            s,
            cur_b,
            zeros_alpha,
            kappas,
            r3f,
            r4f,
            c3b,
            c4b,
            g3f,
            g4f,
            x_evals,
            A_fwd,
            x_end,
        )
        for q in range(n_flag):
            f = flag_idx[q]
            S_plus[q, j] = out[f]
            S_minus[q, j] = -phase_end[f] * out_b[f]
    return S_plus, S_minus


@nb.njit(cache=True)
def _mode_scales(g, floor):
    """Per-mode scale ``max_x |g_j| + floor`` used to weight residuals."""
    n_modes, n_x = g.shape
    out = np.empty(n_modes, dtype=np.float64)
    for j in range(n_modes):
        m = 0.0
        for t in range(n_x):
            a = np.abs(g[j, t])
            if a > m:
                m = a
        out[j] = m + floor
    return out


@nb.njit(cache=True)
def _relative_residual(f, scales):
    """Largest per-mode relative fixed-point residual ``max_x |f_j| / scale_j``.

    Infinite if any entry or scale is not finite: a blown-up iterate must
    never pass for converged (NaN entries would otherwise be skipped by the
    comparisons) nor reach the Anderson least-squares step.
    """
    n_modes, n_x = f.shape
    worst = 0.0
    for j in range(n_modes):
        if not np.isfinite(scales[j]):
            return np.inf
        m = 0.0
        for t in range(n_x):
            a = np.abs(f[j, t])
            if not np.isfinite(a):
                return np.inf
            if a > m:
                m = a
        v = m / scales[j]
        if v > worst:
            worst = v
    return worst


@nb.njit(cache=True)
def _anderson_mix(g, f, dG, dF, n_hist, head, depth, scales):
    """
    Anderson (type II) update ``u = g - sum_a gamma_a dG_a``.

    ``gamma`` minimises the weighted norm ``|| f - sum_a gamma_a dF_a ||``,
    with every mode weighted by ``1 / scales_j`` so that weak modes (signal,
    idlers) count as much as the pump. The small normal-equation system is
    Tikhonov-regularised for robustness.
    """
    n_modes, n_x = g.shape
    w2 = 1.0 / scales**2
    A = np.zeros((n_hist, n_hist), dtype=np.complex128)
    b = np.zeros(n_hist, dtype=np.complex128)
    for a in range(n_hist):
        ia = (head - 1 - a) % depth
        for c in range(a, n_hist):
            ic = (head - 1 - c) % depth
            acc = 0.0 + 0.0j
            for j in range(n_modes):
                s = 0.0 + 0.0j
                for t in range(n_x):
                    s += np.conj(dF[ia, j, t]) * dF[ic, j, t]
                acc += s * w2[j]
            A[a, c] = acc
            A[c, a] = np.conj(acc)
        acc = 0.0 + 0.0j
        for j in range(n_modes):
            s = 0.0 + 0.0j
            for t in range(n_x):
                s += np.conj(dF[ia, j, t]) * f[j, t]
            acc += s * w2[j]
        b[a] = acc
    tr = 0.0
    for a in range(n_hist):
        tr += A[a, a].real
    for a in range(n_hist):
        A[a, a] += 1e-12 * tr + 1e-300
    gam = np.linalg.solve(A, b)
    u = g.copy()
    for a in range(n_hist):
        ia = (head - 1 - a) % depth
        for j in range(n_modes):
            for t in range(n_x):
                u[j, t] -= gam[a] * dG[ia, j, t]
    return u


@nb.njit(parallel=True, cache=True)
def _bloch_fb_solve_uniform(
    x_array,
    y0_array_fwd,
    y0_array_bwd,
    gammas_array,
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
    Zb_plus_table,
    Zb_minus_table,
    passes,
    kappa_cutoff,
    m_lim_table,
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
    edge_K_table,
    edge_flag_table,
    edge_substeps,
):
    """Forward/backward Bloch CME solve for all frequency points (uniform device)."""
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
    converged_flags = np.zeros(n_freq, dtype=np.bool_)
    residuals = np.full(n_freq, np.inf)

    for i in nb.prange(n_freq):
        alphas_i = np.ascontiguousarray(gammas_array[i].real)
        kappas_i = np.ascontiguousarray(gammas_array[i].imag)
        Zp = Zb_plus_table[i]
        Zm = Zb_minus_table[i]
        Ze_i = Ze_table[i]
        Zout_in = Zout_in_table[i]
        Zout_out = Zout_out_table[i]

        # Exact boundary coefficients in the Bloch basis (constant).
        t_in = 2.0 * Ze_i / (apcz_in_table[i] * (Zp + Zout_in))
        rho_in = (Zm - Zout_in) / (Zp + Zout_in)
        t_out = 2.0 * Ze_i / (apcz_out_table[i] * (Zm + Zout_out))
        rho_out = (Zp - Zout_out) / (Zm + Zout_out)
        tof_in = tof_in_table[i]
        tof_out_signed = -tof_out_table[i]  # backward current referenced to +x

        phase_end = np.exp(1j * kappas_i * x_end)

        r3, r4 = _filter_wm_terms_bloch(
            kappas_i,
            relations_3wm,
            relations_4wm,
            kappa_cutoff,
            n_cells,
            m_lim_table[i],
        )
        g_unit = 2.0 * np.pi / n_cells
        g3 = np.empty(r3.shape[0], dtype=np.float64)
        for r in range(r3.shape[0]):
            g3[r] = r3[r, 4] * g_unit
        g4 = np.empty(r4.shape[0], dtype=np.float64)
        for r in range(r4.shape[0]):
            g4[r] = r4[r, 5] * g_unit
        phi_p = np.ascontiguousarray(phi_plus_table[i])
        phi_m = np.ascontiguousarray(phi_minus_table[i])
        prefac = np.ascontiguousarray(prefac_table[i])
        c3_fwd = _bloch_row_coeffs(
            r3, base_3wm, prefac, phi_p, phi_m, L_nl, positions, n_cells, True
        )
        c3_bwd = _bloch_row_coeffs(
            r3, base_3wm, prefac, phi_p, phi_m, L_nl, positions, n_cells, False
        )
        c4_fwd = _bloch_row_coeffs(
            r4, base_4wm, prefac, phi_p, phi_m, L_nl, positions, n_cells, True
        )
        c4_bwd = _bloch_row_coeffs(
            r4, base_4wm, prefac, phi_p, phi_m, L_nl, positions, n_cells, False
        )
        # Stopband-edge guard: one real factor per process (see
        # twpasolver.cmes.bloch.process_weights), applied through the base
        # relation index so all direction / G variants share it.
        for r in range(r3.shape[0]):
            wgt = pw3_table[i, r3[r, 3]]
            c3_fwd[r] *= wgt
            c3_bwd[r] *= wgt
        for r in range(r4.shape[0]):
            wgt = pw4_table[i, r4[r, 4]]
            c4_fwd[r] *= wgt
            c4_bwd[r] *= wgt

        I_fwd = t_in * y0_array_fwd[i]
        I_bwd = np.zeros(n_modes, dtype=np.complex128)
        A_bwd_i = np.zeros((n_modes, n_x), dtype=np.complex128)

        # Edge-basis modes: predictor in the forward sweep (code 2), frozen
        # in the backward sweep (code 1), exact 2x2 solve after both.
        is_flag = np.ascontiguousarray(edge_flag_table[i])
        n_flag = 0
        for j in range(n_modes):
            if is_flag[j] > 0.0:
                n_flag += 1
        flag_idx = np.empty(n_flag, dtype=np.int64)
        code_fwd = np.zeros(n_modes, dtype=np.float64)
        code_bwd = np.zeros(n_modes, dtype=np.float64)
        lin_own = np.zeros(n_modes, dtype=np.complex128)
        lin_ctr = np.zeros(n_modes, dtype=np.complex128)
        q = 0
        for j in range(n_modes):
            if is_flag[j] > 0.0:
                flag_idx[q] = j
                code_fwd[j] = 2.0
                code_bwd[j] = 1.0
                lin_own[j] = edge_K_table[i, j, 0, 0]
                lin_ctr[j] = edge_K_table[i, j, 0, 1]
                q += 1
        n_sub = int(np.ceil(x_end / n_cells * edge_substeps)) + 1
        xs_edge = np.linspace(x_array0[0], x_end, n_sub)
        h_edge = xs_edge[1] - xs_edge[0]
        own_bwd = np.zeros((n_modes, n_x), dtype=np.complex128)

        converged = False
        stable_count = 0
        prev_fwd = np.zeros(n_modes, dtype=np.complex128)
        prev_bwd = np.zeros(n_modes, dtype=np.complex128)

        # Anderson acceleration of the pass map u -> G(u), with u the backward
        # envelopes fed to the forward solve (see module docstring).
        use_aa = anderson_depth > 0
        depth = anderson_depth if use_aa else 1
        dG = np.zeros((depth, n_modes, n_x), dtype=np.complex128)
        dF = np.zeros((depth, n_modes, n_x), dtype=np.complex128)
        g_last = np.zeros((n_modes, n_x), dtype=np.complex128)
        f_last = np.zeros((n_modes, n_x), dtype=np.complex128)
        n_hist = 0
        head = 0
        have_last = False
        res_last = np.inf
        scale_floor = 10.0 * SOLVER_ATOL

        u_k = A_bwd_i
        for k in range(passes):
            depth = min(anderson_depth, k + 1) if use_aa else 1
            for q in range(n_flag):
                own_bwd[flag_idx[q]] = u_k[flag_idx[q]]

            A_fwd_i = rk45_solve(
                bloch_fb_rhs_edge,
                x_span,
                I_fwd,
                args=(
                    alphas_i,
                    kappas_i,
                    r3,
                    r4,
                    c3_fwd,
                    c4_fwd,
                    g3,
                    g4,
                    x_array0,
                    u_k,
                    x_end,
                    code_fwd,
                    own_bwd,
                    lin_own,
                    lin_ctr,
                    phase_end,
                ),
                atol=SOLVER_ATOL,
                rtol=SOLVER_RTOL,
                max_num_steps=SOLVER_MAX_STEPS,
                t_eval=x_array0,
                first_step=SOLVER_FIRST_STEP,
                max_step=max_step,
            )

            for j in range(n_modes):
                I_bwd[j] = (
                    -t_out[j] * y0_array_bwd[i, j]
                    + rho_out[j] * A_fwd_i[j, -1] * phase_end[j]
                )

            A_bwd_i = rk45_solve(
                bloch_fb_rhs_edge,
                x_span,
                I_bwd,
                args=(
                    alphas_i,
                    kappas_i,
                    r3,
                    r4,
                    c3_bwd,
                    c4_bwd,
                    g3,
                    g4,
                    x_array0,
                    A_fwd_i,
                    x_end,
                    code_bwd,
                    own_bwd,
                    lin_own,
                    lin_ctr,
                    phase_end,
                ),
                atol=SOLVER_ATOL,
                rtol=SOLVER_RTOL,
                max_num_steps=SOLVER_MAX_STEPS,
                t_eval=x_array0,
                first_step=SOLVER_FIRST_STEP,
                max_step=max_step,
            )

            for q in range(n_flag):
                A_bwd_i[flag_idx[q]] = own_bwd[flag_idx[q]]
            if n_flag > 0:
                S_p, S_m = _edge_sources(
                    xs_edge,
                    flag_idx,
                    is_flag,
                    kappas_i,
                    r3,
                    r4,
                    c3_fwd,
                    c4_fwd,
                    c3_bwd,
                    c4_bwd,
                    g3,
                    g4,
                    x_array0,
                    A_fwd_i,
                    A_bwd_i,
                    x_end,
                    phase_end,
                )
                for q in range(n_flag):
                    f = flag_idx[q]
                    pe = phase_end[f]
                    c_p, c_m = _edge_bvp_solve(
                        edge_K_table[i, f],
                        h_edge,
                        S_p[q],
                        S_m[q],
                        t_in[f] * y0_array_fwd[i, f],
                        rho_in[f],
                        -t_out[f] * y0_array_bwd[i, f] * pe,
                        rho_out[f] * pe * pe,
                    )
                    # Back to the solver grid: c+ in x, B(s) = c-(L - s) / pe.
                    for t in range(n_x):
                        A_fwd_i[f, t] = np.interp(
                            x_array0[t], xs_edge, c_p.real
                        ) + 1j * np.interp(x_array0[t], xs_edge, c_p.imag)
                        xm = x_end - x_array0[t]
                        A_bwd_i[f, t] = (
                            np.interp(xm, xs_edge, c_m.real)
                            + 1j * np.interp(xm, xs_edge, c_m.imag)
                        ) / pe

            # Fixed-point residual of the pass map: A_bwd_i is G(u), u is the
            # backward field the forward solve just used. Each backward
            # envelope is judged against its own and its forward partner's
            # amplitude: weak reflections need not be resolved beyond what
            # they feed back.
            g_ext = A_bwd_i
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
                        # Restart: the history no longer describes the map.
                        n_hist = 0
                        have_last = False
                    if have_last:
                        dG[head] = g_ext - g_last
                        dF[head] = f_k - f_last
                        head = (head + 1) % depth
                        if n_hist < depth:
                            n_hist += 1
                    g_last[:, :] = g_ext
                    f_last[:, :] = f_k
                    have_last = True
                    res_last = res
                    if n_hist > 0:
                        u_next = _anderson_mix(
                            g_ext, f_k, dG, dF, n_hist, head, depth, scales
                        )
                    else:
                        u_next = g_ext.copy()
                else:
                    u_next = g_ext.copy()
            else:
                u_next = g_ext
                if k > 0:
                    stable_count, converged = _convergence_step(
                        A_fwd_i,
                        A_bwd_i,
                        prev_fwd,
                        prev_bwd,
                        n_modes,
                        convergence_threshold,
                        stable_count,
                    )

            for j in range(n_modes):
                I_fwd[j] = (
                    t_in[j] * y0_array_fwd[i, j]
                    + rho_in[j] * u_next[j, -1] * phase_end[j]
                )

            id_add_fwd = 0
            id_add_bwd = n_freq
            if save_all_passes:
                id_add_fwd = k * n_freq
                id_add_bwd = k * n_freq + n_freq * passes
            _store_pass(
                I_out,
                A_fwd_i,
                A_bwd_i,
                rho_out,
                rho_in,
                tof_out_signed,
                tof_in,
                i + id_add_fwd,
                i + id_add_bwd,
                n_modes,
            )

            for j in range(n_modes):
                prev_fwd[j] = A_fwd_i[j, -1]
                prev_bwd[j] = A_bwd_i[j, -1]
            u_k = np.ascontiguousarray(u_next)

            if converged:
                converged_flags[i] = True
                if save_all_passes:
                    for kk in range(k + 1, passes):
                        _store_pass(
                            I_out,
                            A_fwd_i,
                            A_bwd_i,
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


def bloch_fb_cmes_solve(
    x_array: FloatArray,
    y0_array_fwd: ComplexArray,
    y0_array_bwd: ComplexArray,
    gammas_array: ComplexArray,
    relations_3wm: np.ndarray,
    relations_4wm: np.ndarray,
    base_3wm: ComplexArray,
    base_4wm: ComplexArray,
    prefac: ComplexArray,
    phi_plus: ComplexArray,
    phi_minus: ComplexArray,
    L_nl: FloatArray,
    positions: FloatArray,
    n_cells: int,
    Zb_plus: ComplexArray,
    Zb_minus: ComplexArray,
    passes: int = 10,
    kappa_cutoff: float = 0.2,
    m_max: int | np.ndarray = 5,
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
    edge_K: ComplexArray | None = None,
    edge_flag: np.ndarray | None = None,
    edge_substeps: int = 4,
    diagnostics: dict | None = None,
) -> ComplexArray:
    """
    Solve the forward/backward Bloch CMEs for every signal frequency.

    Parameters
    ----------
    gammas_array : (n_freq, n_modes)
        Kernel propagation constants (``-alpha - j k``); uniform devices only.
    base_3wm, base_4wm : (n_rel,)
        RWA multiplicity/4 times epsilon / xi, aligned with ``relations_*``.
    prefac : (n_freq, n_modes)
        ``-2j omega / W``.
    phi_plus, phi_minus : (n_freq, n_modes, n_nl)
        Bloch current profiles through the nonlinear inductors.
    L_nl, positions : (n_nl,)
        Nonlinear inductances and their positions in the supercell (unit cells).
    Zb_plus, Zb_minus : (n_freq, n_modes)
        Bloch impedances of the forward/backward modes at the device ends.
    kappa_cutoff : float
        Relative phase-mismatch cutoff, ``|dk + G_m| < kappa_cutoff * k_ref``
        with ``k_ref = min(k_proc, pi / d)``: ``k_proc`` the largest wavenumber
        among the modes of the process, capped at the zone edge.
    m_max : int or (n_freq, n_modes) int array
        Largest ``|m|`` of the reciprocal vectors ``G_m = 2 pi m / d`` of the
        intracell coupling profiles (quasi-phase matching); 0 keeps only the
        cell averages. An array gives a limit per frequency and mode; a
        process uses the largest limit among its modes.
    anderson_depth : int
        Number of previous passes used by Anderson acceleration of the
        forward/backward iteration; 0 gives the plain fixed-point iteration.
        The iterate is the backward envelope profile fed to the forward solve;
        convergence is declared when the largest per-mode relative residual
        ``max|G(u) - u| / (max|G(u)| + max|A+|)`` is below
        ``convergence_threshold`` (each backward envelope is measured against
        its own and its forward partner's amplitude).
        Anderson acceleration also converges when the plain iteration
        diverges because a few eigenvalues of the pass map lie outside the
        unit circle (strong counter-propagating coupling).
    process_weights_3wm, process_weights_4wm : (n_freq, n_relations), optional
        Real factor of every relation row (stopband-edge guard, see
        :func:`twpasolver.cmes.bloch.process_weights`); default 1.
    Ze, Zout_in, Zout_out, abcd_in, abcd_out :
        Termination description, same meaning as in
        ``general_cmes_solve_fb_cutoff``.
    max_step : float
        Largest integrator step (unit cells). The integrator returns the
        envelopes on ``x_array`` by linear interpolation between accepted
        steps; capping the step at the grid spacing keeps the stored fields
        (read back as counter-propagating fields, or by the quantum model)
        accurate when they rotate under self/cross-phase modulation.
        Default: no cap (adaptive steps only).
    edge_K : (n_freq, n_modes, 2, 2) complex, optional
        Per-unit-cell linear generator ``K`` of the edge-basis amplitudes
        ``(c+, c-)`` (see :func:`twpasolver.cmes.bloch.edge_basis_tables`),
        used where ``edge_flag`` is set.
    edge_flag : (n_freq, n_modes) bool, optional
        Modes expanded on a fixed edge basis. Their Bloch tables, ``prefac``
        and kernel gamma (``-1j * kappa_e``) must already be the edge-basis
        ones. ``c+`` is predicted in the forward sweep, ``c-`` frozen in the
        backward one, and both are replaced by the exact 2x2 solve once per
        pass (module notes above). Default: none.
    edge_substeps : int
        Grid points per supercell of the exact edge-mode solve.
    diagnostics : dict, optional
        If given, filled with ``converged`` (n_freq,) bool and ``residuals``
        (n_freq,) float, the last relative fixed-point residual.

    Returns
    -------
    ndarray, shape (2 * n_freq [* passes], n_modes, n_x)
        Same layout as ``general_cmes_solve_fb_cutoff``: forward envelopes
        first, then backward envelopes (on the mirrored coordinate); the last
        x-point holds the current coupled out to the external line.
    """
    gammas_array = np.asarray(gammas_array)
    if gammas_array.ndim != 2:
        raise NotImplementedError(
            "Bloch forward/backward CMEs support only non-segmented devices."
        )
    n_freq, n_modes = gammas_array.shape

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
            return np.ones((n_freq, n_rel), dtype=np.float64)
        arr = np.asarray(val, dtype=np.float64)
        if arr.shape != (n_freq, n_rel):
            raise ValueError(
                f"Process weights must have shape {(n_freq, n_rel)}, got {arr.shape}."
            )
        return arr

    pw3 = _weights(process_weights_3wm, len(relations_3wm))
    pw4 = _weights(process_weights_4wm, len(relations_4wm))

    if edge_flag is None:
        edge_flag_f = np.zeros((n_freq, n_modes), dtype=np.float64)
    else:
        edge_flag_f = np.asarray(edge_flag, dtype=np.float64)
    if edge_K is None:
        edge_K = np.zeros((n_freq, n_modes, 2, 2), dtype=np.complex128)

    c = np.ascontiguousarray
    I_out, converged, residuals = _bloch_fb_solve_uniform(
        c(x_array, dtype=np.float64),
        c(y0_array_fwd, dtype=np.complex128),
        c(y0_array_bwd, dtype=np.complex128),
        c(gammas_array, dtype=np.complex128),
        c(relations_3wm, dtype=np.int64),
        c(relations_4wm, dtype=np.int64),
        c(base_3wm, dtype=np.complex128),
        c(base_4wm, dtype=np.complex128),
        c(prefac, dtype=np.complex128),
        c(phi_plus, dtype=np.complex128),
        c(phi_minus, dtype=np.complex128),
        c(L_nl, dtype=np.float64),
        c(positions, dtype=np.float64),
        int(n_cells),
        c(Zb_plus, dtype=np.complex128),
        c(Zb_minus, dtype=np.complex128),
        int(passes),
        float(kappa_cutoff),
        c(np.broadcast_to(np.asarray(m_max, dtype=np.int64), gammas_array.shape)),
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
        c(edge_K, dtype=np.complex128),
        c(edge_flag_f),
        int(edge_substeps),
    )
    if not np.all(converged):
        bad = np.flatnonzero(~converged)
        log.warning(
            "Bloch forward-backward CMEs did not converge for %d of %d frequency "
            "points within %d passes (indices %s; last relative residuals %s). "
            "Residuals around 1e-3-1e-1 that stop decreasing usually mean the "
            "integrator noise floor of very weak modes (raise Is0 or the "
            "convergence threshold); large or non-finite residuals mean the "
            "iteration failed, typically because a mode lies close to a "
            "stopband edge.",
            bad.size,
            n_freq,
            passes,
            bad[:20].tolist(),
            np.array2string(
                residuals[bad[:20]], formatter={"float_kind": lambda v: f"{v:.1e}"}
            ),
        )
    if diagnostics is not None:
        diagnostics["converged"] = converged
        diagnostics["residuals"] = residuals
    return I_out
