"""
Numba helpers shared by the forward-backward Bloch CME solver.

* cubic interpolation of stored envelopes (``_cubic_stencil``,
  ``_stencil_value``, ``_interp_rows``);
* selection of the phase-matched mixing terms with their reciprocal vectors
  (``_filter_wm_terms_bloch``) and their Bloch coefficients
  (``_bloch_row_coeffs``, ``_rows_with_targets``);
* closed-form exponential of a 2x2 generator (``_expm_2x2``);
* fixed-point iteration: residual scales, relative residual, Anderson mixing
  (``_mode_scales``, ``_relative_residual``, ``_anderson_mix``).
"""

import numba as nb
import numpy as np

from twpasolver.cmes.base import _sorted_interp_bracket

# --------------------------------------------------------------------------- #
# Interpolation of stored envelopes
# --------------------------------------------------------------------------- #


@nb.njit(cache=True)
def _cubic_stencil(xp, x):
    """
    First index and weights of the 4-point Lagrange stencil around ``x``.

    The stored envelopes oscillate with the residual mismatch of the kept
    terms, up to ``kappa_cutoff * pi`` per supercell; for a phase ``phi`` per
    grid step the cubic lookup error is about ``phi^4 / 50`` (linear:
    ``phi^2 / 8``). Outside the grid the value is clamped as in ``np.interp``;
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
def _interp_rows(arr, x_evals, x):
    """Cubic interpolation of every row of ``arr`` (rows, n_x) at ``x``."""
    i0, w0, w1, w2, w3 = _cubic_stencil(x_evals, x)
    out = np.empty(arr.shape[0], dtype=np.complex128)
    for r in range(arr.shape[0]):
        out[r] = _stencil_value(arr, r, i0, w0, w1, w2, w3)
    return out


# --------------------------------------------------------------------------- #
# Mixing terms and their Bloch coefficients
# --------------------------------------------------------------------------- #


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

    The coefficient of a row with index ``m`` is the Fourier component
    ``(1/d) sum_j F_j exp(-i G x_j)`` of its intracell profile, with
    ``G = 2 pi m / d`` in the solve coordinate. The backward solve runs in the
    mirrored coordinate ``s = L - x``, where ``exp(i G s)`` corresponds to the
    physical component ``-G``, hence the sign flip of the phase.

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


# --------------------------------------------------------------------------- #
# Linear algebra
# --------------------------------------------------------------------------- #


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


# --------------------------------------------------------------------------- #
# Fixed-point iteration
# --------------------------------------------------------------------------- #


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
    idlers) count as much as the pump. The normal equations are
    Tikhonov-regularised.
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
