"""
Forward-backward coupled mode equations in the Bloch-mode basis.

Extension of :mod:`twpasolver.cmes.bloch` to counter-propagating waves,
reflections at the device terminations and segmented (piecewise-uniform)
devices. A segmented device is a chain of uniform segments, each made of whole
supercells of its own; a uniform device is the one-segment case of
:func:`bloch_fb_segmented_cmes_solve`.

Physics summary (package convention exp(+j omega t), forward ~ exp(-gamma x))
-----------------------------------------------------------------------------
* Inside a segment every mode is expanded exactly as
  ``psi_n = A+_n psi+_n + A-_n psi-_n`` on that segment's Bloch modes, with
  ``psi+_n(0) = (Zb+_n, 1)`` and ``psi-_n(0) = (-Zb-_n, 1)``. Both amplitudes
  are currents referenced to +x, so the total current through a nonlinear
  inductor is ``A+ phi+ e^{-gamma x} + A- phi- e^{+gamma x}``.
* Projecting the nonlinear series source with the reciprocity invariant
  ``W_n = <psi+_n, psi-_n> = Zb+_n + Zb-_n`` gives
  ``dA+_n/dx = -(j omega_n / W_n) <L Pi_n phi-_n> e^{+gamma x}`` and
  ``dA-_n/dx = +(j omega_n / W_n) <L Pi_n phi+_n> e^{-gamma x}``.
  The backward solve runs in the mirrored coordinate s = L - x, which flips
  the sign of the second equation: both solves use the prefactor
  ``-2j omega_n / W_n`` (times the RWA multiplicity/4 and epsilon or xi),
  and differ only in the partner profile (phi- for the forward target, phi+
  for the backward one) and in which profile each input carries
  (own direction vs counter direction). Reciprocal vectors ``2 pi m / d_s``,
  term selection and stopband-edge treatment are those of each segment.
* Because the expansion fixes voltage and current at every supercell
  boundary (``phi = 1`` there), ``V = Zb+ A+ - Zb- A-`` and ``I = A+ + A-``
  hold exactly, with the nonlinearity on, at the device ends and at every
  junction. The boundary coefficients are therefore constant (no derivative
  correction, no damping of a self-consistent update):

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

Junction conditions
-------------------
Every junction lies on a supercell boundary of both neighbours. Continuity of
(V, I) and the projection with the reciprocity invariant give, for the junction
between a left segment L and a right segment R::

    A+_R = t  A+_L + rho_R A-_R,      t   = W_L / (Zb+_R + Zb-_L),
    A-_L = t' A-_R + rho_L A+_L,      t'  = W_R / (Zb+_R + Zb-_L),
                                      rho_R = (Zb-_R - Zb-_L) / (Zb+_R + Zb-_L),
                                      rho_L = (Zb+_L - Zb+_R) / (Zb+_R + Zb-_L).

The input boundary above is the special case ``Zb-_L -> Zout``. Identical
neighbours give ``t = t' = 1``, ``rho = 0``. In flux variables
(``a ∝ sqrt(W) A``) the junction map is reciprocal, and unitary for lossless
passbands, so energy and Manley-Rowe conservation survive composition.

Phases and coordinates
----------------------
Only the propagation phase is factored out of the integration variables; the
loss stays in the envelope through the ``alpha * A`` term. With the
piecewise-constant wavenumber the phase is

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
forward fields just computed). The pass is a map u -> G(u) on the backward
envelopes u on the internal grid. Strong counter-propagating coupling (e.g. a
Bragg-matched forward/backward pair near a stopband, or internal reflections)
puts eigenvalues of this map outside the unit circle and the plain iteration
diverges even though the boundary-value problem has a regular solution.
Anderson acceleration (``anderson_depth`` previous passes, per-mode weighted
least squares) solves for the fixed point instead. It cannot cure an
instability inside a single sweep (a mode close to a stopband edge whose
forward-only subsystem grows exponentially); such modes are solved exactly on
a fixed edge basis (``edge_K``, ``edge_mode``).
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
from twpasolver.cmes.bloch_common import (
    _anderson_mix,
    _bloch_row_coeffs,
    _cubic_stencil,
    _expm_2x2,
    _filter_wm_terms_bloch,
    _interp_rows,
    _mode_scales,
    _relative_residual,
    _rows_with_targets,
    _stencil_value,
)
from twpasolver.cmes.rk45 import rk45_solve
from twpasolver.cmes.solver_config import BlochSolverConfig, RKSolverConfig
from twpasolver.logger import log

_DEFAULTS = BlochSolverConfig()

# Grid points per supercell of the exact edge-mode solve.
EDGE_SUBSTEPS = 4

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

    Field bookkeeping as described in the module docstring,
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
    i0, w0, w1, w2, w3 = _cubic_stencil(ctr_x, y)
    for i in range(n):
        ctr = _stencil_value(ctr_vals, i, i0, w0, w1, w2, w3) * en[i]
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


# --------------------------------------------------------------------------- #
# Edge-basis modes in a segmented device
# --------------------------------------------------------------------------- #
#
# Edge modes: predicted in the forward sweep, frozen in the backward sweep,
# solved exactly as a 2x2 two-point problem once per pass, with
# a generator K_s per segment: the fixed-basis K in segments where the mode is
# flagged, diag(-alpha, alpha) in its native segments. With the forward phase
# Phi_f(x) (piecewise linear, continuous) the physical boundary amplitudes are
# c+ exp(j Phi_f) and c- exp(-j Phi_f), so a junction at x_j reads
#     c+_R = t c+_L + rho_R exp(-2j Phi_f(x_j)) c-_R,
#     c-_L = t' c-_R + rho_L exp(+2j Phi_f(x_j)) c+_L,
# and the backward envelope storage is c-(x) = B(L - x) exp(j Phi_f(L)).


@nb.njit(cache=True)
def bloch_fb_segment_rhs_edge(
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
    edge_code,
    own_x,
    own_vals,
    lin_own,
    lin_ctr,
    ctr_phase,
):
    """
    :func:`bloch_fb_segment_rhs` with the edge-basis modes of one solve direction.

    ``edge_code[i]``: 0 ordinary mode; 1 frozen (own field read from
    ``own_vals`` on ``own_x``, zero derivative); 2 integrated with the extra
    linear terms ``lin_own[i] c + lin_ctr[i] ctr_phase[i] u_i(end - x)``, with
    ``u_i`` the counter envelope of the segment (``ctr_vals`` on ``ctr_x``).
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
        i0, w0, w1, w2, w3 = _cubic_stencil(own_x, x)
        for i in range(n):
            if edge_code[i] == 1.0:
                cur[i] = _stencil_value(own_vals, i, i0, w0, w1, w2, w3)
    out = bloch_fb_segment_rhs(
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
        x0,
        phi0,
        ctr_x,
        ctr_vals,
        ctr_y0,
        ctr_phi0,
        end,
    )
    j0, v0, v1, v2, v3 = 0, 0.0, 0.0, 0.0, 0.0
    if any_lin:
        j0, v0, v1, v2, v3 = _cubic_stencil(ctr_x, end - x)
    for i in range(n):
        if edge_code[i] == 1.0:
            out[i] = 0.0
        elif edge_code[i] == 2.0:
            ctr = _stencil_value(ctr_vals, i, j0, v0, v1, v2, v3)
            out[i] += lin_own[i] * cur[i] + lin_ctr[i] * ctr_phase[i] * ctr
    return out


@nb.njit(cache=True)
def _edge_sources_segmented(
    xs,
    sub_off,
    flag_idx,
    is_flag,
    kappas_seg,
    r3,
    r4,
    off3,
    off4,
    g3,
    g4,
    c3f,
    c4f,
    c3b,
    c4b,
    x_int,
    s_int,
    offsets,
    Phi_f_b,
    Phi_b_b,
    A_fwd,
    A_bwd,
    end,
    phase_end,
):
    """
    Nonlinear sources ``S+-(x)`` of the edge modes on the edge grid ``xs``.

    Segment ``s`` owns ``xs[sub_off[s]:sub_off[s+1]]`` (both ends included).
    ``S+`` is the forward-solve derivative of ``c+`` and ``S- = -exp(j Phi_f(L))
    dB/ds`` the physical derivative of ``c-``; only rows targeting an edge mode
    are evaluated, with the segment's own rows, coefficients and phases.
    """
    n_modes = A_fwd.shape[0]
    n_seg = offsets.shape[0] - 1
    n_int = x_int.shape[0]
    n_flag = flag_idx.shape[0]
    S_plus = np.zeros((n_flag, xs.shape[0]), dtype=np.complex128)
    S_minus = np.zeros((n_flag, xs.shape[0]), dtype=np.complex128)
    zeros_alpha = np.zeros(n_modes, dtype=np.float64)
    for s in range(n_seg):
        q = n_seg - 1 - s
        a = offsets[s]
        b = offsets[s + 1]
        ca = n_int - offsets[s + 1]
        cb = n_int - offsets[s]
        r3f, c3ff, g3f = _rows_with_targets(
            r3[off3[s] : off3[s + 1]],
            c3f[off3[s] : off3[s + 1]],
            g3[off3[s] : off3[s + 1]],
            is_flag,
        )
        r4f, c4ff, g4f = _rows_with_targets(
            r4[off4[s] : off4[s + 1]],
            c4f[off4[s] : off4[s + 1]],
            g4[off4[s] : off4[s + 1]],
            is_flag,
        )
        _, c3bf, _ = _rows_with_targets(
            r3[off3[s] : off3[s + 1]],
            c3b[off3[s] : off3[s + 1]],
            g3[off3[s] : off3[s + 1]],
            is_flag,
        )
        _, c4bf, _ = _rows_with_targets(
            r4[off4[s] : off4[s + 1]],
            c4b[off4[s] : off4[s + 1]],
            g4[off4[s] : off4[s + 1]],
            is_flag,
        )
        seg_x = np.ascontiguousarray(x_int[a:b])
        seg_s = np.ascontiguousarray(s_int[ca:cb])
        Af = np.ascontiguousarray(A_fwd[:, a:b])
        Ab = np.ascontiguousarray(A_bwd[:, ca:cb])
        k_s = np.ascontiguousarray(kappas_seg[s])
        pf = np.ascontiguousarray(Phi_f_b[:, s])
        pb = np.ascontiguousarray(Phi_b_b[:, q])
        for j in range(sub_off[s], sub_off[s + 1]):
            x = xs[j]
            cur = _interp_rows(Af, seg_x, x)
            out = bloch_fb_segment_rhs(
                x,
                cur,
                zeros_alpha,
                k_s,
                r3f,
                r4f,
                c3ff,
                c4ff,
                g3f,
                g4f,
                seg_x[0],
                pf,
                seg_s,
                Ab,
                seg_s[0],
                pb,
                end,
            )
            y = end - x
            cur_b = _interp_rows(Ab, seg_s, y)
            out_b = bloch_fb_segment_rhs(
                y,
                cur_b,
                zeros_alpha,
                k_s,
                r3f,
                r4f,
                c3bf,
                c4bf,
                g3f,
                g4f,
                seg_s[0],
                pb,
                seg_x,
                Af,
                seg_x[0],
                pf,
                end,
            )
            for qq in range(n_flag):
                f = flag_idx[qq]
                S_plus[qq, j] = out[f]
                S_minus[qq, j] = -phase_end[f] * out_b[f]
    return S_plus, S_minus


@nb.njit(cache=True)
def _edge_bvp_solve_segmented(
    K_seg,
    xs,
    sub_off,
    S_plus,
    S_minus,
    cp_in,
    rho_in,
    cm_out,
    rho_out,
    t_j,
    tp_j,
    rhoL_j,
    rhoR_j,
    Phi_j,
):
    """
    Exact two-point solution of ``dc/dx = K_s c + S`` across segments and junctions.

    Inside segment ``s`` the step is ``exp(K_s h_s)`` with trapezoidal sources
    (uniform sub-grid ``xs[sub_off[s]:sub_off[s+1]]``, both ends included). The
    discrete Riccati sweep ``c-_j = R_j c+_j + Q_j`` is built from the output
    end, ``c-(L) = cm_out + rho_out c+(L)``, and carried across junction ``s``
    (between segments ``s-1`` and ``s``) by its scattering relations
    (``t_j``, ``tp_j``, ``rhoL_j``, ``rhoR_j`` of the mode, forward phase
    ``Phi_j[s]`` there); the forward pass starts from
    ``c+(0) = cp_in + rho_in c-(0)``. Returns ``(c_plus, c_minus)`` on ``xs``,
    with the left and right limits at each junction in the two segments.
    """
    n_seg = K_seg.shape[0]
    n = xs.shape[0]
    P = np.empty((n_seg, 2, 2), dtype=np.complex128)
    sp = np.zeros(n, dtype=np.complex128)
    sm = np.zeros(n, dtype=np.complex128)
    for s in range(n_seg):
        lo = sub_off[s]
        hi = sub_off[s + 1]
        h = xs[lo + 1] - xs[lo]
        P[s] = _expm_2x2(K_seg[s] * h)
        a, b, c, d = P[s, 0, 0], P[s, 0, 1], P[s, 1, 0], P[s, 1, 1]
        for j in range(lo, hi - 1):
            sp[j] = 0.5 * h * (a * S_plus[j] + b * S_minus[j] + S_plus[j + 1])
            sm[j] = 0.5 * h * (c * S_plus[j] + d * S_minus[j] + S_minus[j + 1])
    R = np.empty(n, dtype=np.complex128)
    Q = np.empty(n, dtype=np.complex128)
    R[n - 1] = rho_out
    Q[n - 1] = cm_out
    for s in range(n_seg - 1, -1, -1):
        lo = sub_off[s]
        hi = sub_off[s + 1]
        a, b, c, d = P[s, 0, 0], P[s, 0, 1], P[s, 1, 0], P[s, 1, 1]
        for j in range(hi - 2, lo - 1, -1):
            den = d - R[j + 1] * b
            R[j] = (R[j + 1] * a - c) / den
            Q[j] = (R[j + 1] * sp[j] + Q[j + 1] - sm[j]) / den
        if s > 0:
            e2 = np.exp(2j * Phi_j[s])
            r_r = rhoR_j[s] / e2
            r_l = rhoL_j[s] * e2
            den = 1.0 - r_r * R[lo]
            R[lo - 1] = r_l + tp_j[s] * R[lo] * t_j[s] / den
            Q[lo - 1] = tp_j[s] * Q[lo] / den
    c_plus = np.empty(n, dtype=np.complex128)
    c_minus = np.empty(n, dtype=np.complex128)
    c_plus[0] = (cp_in + rho_in * Q[0]) / (1.0 - rho_in * R[0])
    c_minus[0] = R[0] * c_plus[0] + Q[0]
    for s in range(n_seg):
        lo = sub_off[s]
        hi = sub_off[s + 1]
        a, b = P[s, 0, 0], P[s, 0, 1]
        if s > 0:
            r_r = rhoR_j[s] / np.exp(2j * Phi_j[s])
            den = 1.0 - r_r * R[lo]
            c_plus[lo] = (t_j[s] * c_plus[lo - 1] + r_r * Q[lo]) / den
            c_minus[lo] = R[lo] * c_plus[lo] + Q[lo]
        for j in range(lo, hi - 1):
            c_plus[j + 1] = a * c_plus[j] + b * c_minus[j] + sp[j]
            c_minus[j + 1] = R[j + 1] * c_plus[j + 1] + Q[j + 1]
    return c_plus, c_minus


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
    m_lim_seg,
):
    """
    Kept rows and their coefficients for every segment, concatenated.

    ``m_lim_seg`` (n_seg, n_modes): harmonic limit of every mode per segment
    (see :func:`twpasolver.cmes.bloch_common._filter_wm_terms_bloch`).

    Returns ``r3, r4`` (rows), ``off3, off4`` (per-segment offsets),
    ``g3, g4`` (reciprocal vectors) and the coefficients of both solve
    directions ``c3f, c3b, c4f, c4b``.
    """
    n_seg = kappas_seg.shape[0]
    n_g = 2 * np.max(m_lim_seg) + 1
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
            np.ascontiguousarray(m_lim_seg[s]),
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
def _sweep(
    I0,
    ctr_field,
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
    own_grid,
    ctr_grid,
    own_off,
    ctr_off,
    Phi_own,
    Phi_ctr,
    t_jn,
    rho_jn,
    backward,
    end,
    rk,
    edge_code,
    own_full,
    lin_own_seg,
    lin_ctr_seg,
    ctr_phase,
):
    """
    One sweep over all segments, the counter-propagating ``ctr_field`` frozen.

    Forward: own grid ``x_int`` with ``offsets``, junction coefficients
    ``t, rho_R``; backward: the mirrored coordinate ``s_int`` with
    ``n_int - offsets[::-1]``, junction coefficients ``t', rho_L``. ``p`` runs
    over the segments in the sweep direction, ``c = n_seg - 1 - p`` is the
    same segment in the counter direction. ``Phi_own[m, p]`` / ``Phi_ctr[m, c]``
    are the accumulated phases at segment starts, length ``n_seg + 1``.
    """
    n_modes = I0.shape[0]
    n_seg = own_off.shape[0] - 1
    n_int = own_grid.shape[0]
    A = np.empty((n_modes, n_int), dtype=np.complex128)
    y0 = I0.copy()
    for p in range(n_seg):
        c = n_seg - 1 - p
        s = c if backward else p  # physical segment
        a = own_off[p]
        b = own_off[p + 1]
        ca = ctr_off[c]
        cb = ctr_off[c + 1]
        if p > 0:
            # Forward, junction s at the left of s: A+_R = t A+_L + rho_R A-_R.
            # Backward, junction s + 1 at the right of s: A-_L = t' A-_R + rho_L A+_L.
            jn = s + 1 if backward else s
            for j in range(n_modes):
                A_ctr = ctr_field[j, cb - 1] * np.exp(1j * Phi_ctr[j, c + 1])
                y0[j] = t_jn[jn, j] * A[j, a - 1] + rho_jn[jn, j] * A_ctr * np.exp(
                    -1j * Phi_own[j, p]
                )
        phi0 = np.ascontiguousarray(Phi_own[:, p])
        ctr_phi0 = np.ascontiguousarray(Phi_ctr[:, c])
        seg_x = np.ascontiguousarray(own_grid[a:b])
        sol = rk45_solve(
            bloch_fb_segment_rhs_edge,
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
                np.ascontiguousarray(ctr_grid[ca:cb]),
                np.ascontiguousarray(ctr_field[:, ca:cb]),
                ctr_grid[ca],
                ctr_phi0,
                end,
                edge_code,
                seg_x,
                np.ascontiguousarray(own_full[:, a:b]),
                np.ascontiguousarray(lin_own_seg[s]),
                np.ascontiguousarray(lin_ctr_seg[s]),
                ctr_phase,
            ),
            atol=rk[0],
            rtol=rk[1],
            t_eval=seg_x,
            first_step=rk[2],
            max_steps_per_cell=rk[3],
        )
        A[:, a:b] = sol
    return A


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
    rk,
    edge_K_table,
    edge_mode_table,
):
    """Forward/backward Bloch CME solve for all frequency points (segmented)."""
    n_freq = gammas_seg.shape[0]
    n_seg = gammas_seg.shape[1]
    n_modes = gammas_seg.shape[2]
    n_int = x_int.shape[0]
    n_x = take.shape[0]
    end = seg_bounds[n_seg]
    offsets_b = n_int - offsets[::-1]  # segment offsets on the mirrored s_int

    if save_all_passes:
        I_out = np.empty((2 * n_freq * passes, n_modes, n_x), dtype=np.complex128)
    else:
        I_out = np.empty((2 * n_freq, n_modes, n_x), dtype=np.complex128)
    converged_flags = np.zeros(n_freq, dtype=np.bool_)
    residuals = np.full(n_freq, np.inf)

    for i in nb.prange(n_freq):  # pylint: disable=not-an-iterable
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
            m_lim_table[i],
        )

        I_fwd = t_in * y0_array_fwd[i]
        I_bwd = np.zeros(n_modes, dtype=np.complex128)
        A_bwd_i = np.zeros((n_modes, n_int), dtype=np.complex128)

        # Edge-basis modes: predictor in the forward sweep (code 2), frozen in
        # the backward sweep (code 1), exact 2x2 solve across all segments and
        # junctions after both. Their attenuation is in K, not in the kernel.
        is_flag = np.ascontiguousarray(edge_mode_table[i])
        n_flag = 0
        for j in range(n_modes):
            if is_flag[j] > 0.0:
                n_flag += 1
        flag_idx = np.empty(n_flag, dtype=np.int64)
        code_fwd = np.zeros(n_modes, dtype=np.float64)
        code_bwd = np.zeros(n_modes, dtype=np.float64)
        lin_own_seg = np.zeros((n_seg, n_modes), dtype=np.complex128)
        lin_ctr_seg = np.zeros((n_seg, n_modes), dtype=np.complex128)
        alphas_run = alphas_seg.copy()
        q = 0
        for j in range(n_modes):
            if is_flag[j] > 0.0:
                flag_idx[q] = j
                code_fwd[j] = 2.0
                code_bwd[j] = 1.0
                for s in range(n_seg):
                    alphas_run[s, j] = 0.0
                    lin_own_seg[s, j] = edge_K_table[i, s, j, 0, 0]
                    lin_ctr_seg[s, j] = edge_K_table[i, s, j, 0, 1]
                q += 1
        sub_off = np.zeros(n_seg + 1, dtype=np.int64)
        for s in range(n_seg):
            ln = seg_bounds[s + 1] - seg_bounds[s]
            sub_off[s + 1] = (
                sub_off[s] + int(np.ceil(ln / n_cells[s] * EDGE_SUBSTEPS)) + 1
            )
        xs_edge = np.empty(sub_off[n_seg], dtype=np.float64)
        for s in range(n_seg):
            xs_edge[sub_off[s] : sub_off[s + 1]] = np.linspace(
                seg_bounds[s], seg_bounds[s + 1], sub_off[s + 1] - sub_off[s]
            )
        own_bwd = np.zeros((n_modes, n_int), dtype=np.complex128)

        converged = False
        depth = anderson_depth
        dG = np.zeros((depth, n_modes, n_int), dtype=np.complex128)
        dF = np.zeros((depth, n_modes, n_int), dtype=np.complex128)
        g_last = np.zeros((n_modes, n_int), dtype=np.complex128)
        f_last = np.zeros((n_modes, n_int), dtype=np.complex128)
        n_hist = 0
        head = 0
        have_last = False
        res_last = np.inf
        scale_floor = 10.0 * rk[0]

        A_fwd_x = np.empty((n_modes, n_x), dtype=np.complex128)
        A_bwd_x = np.empty((n_modes, n_x), dtype=np.complex128)

        u_k = A_bwd_i
        for k in range(passes):
            for qq in range(n_flag):
                own_bwd[flag_idx[qq]] = u_k[flag_idx[qq]]
            A_fwd_i = _sweep(
                I_fwd,
                u_k,
                alphas_run,
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
                offsets_b,
                Phi_f_b,
                Phi_b_b,
                t_j,
                rhoR_j,
                False,
                end,
                rk,
                code_fwd,
                own_bwd,
                lin_own_seg,
                lin_ctr_seg,
                phase_end,
            )

            for j in range(n_modes):
                I_bwd[j] = (
                    -t_out[j] * y0_array_bwd[i, j]
                    + rho_out[j] * A_fwd_i[j, n_int - 1] * phase_end[j]
                )

            A_bwd_i = _sweep(
                I_bwd,
                A_fwd_i,
                alphas_run,
                kappas_seg,
                r3,
                r4,
                off3,
                off4,
                g3,
                g4,
                c3b,
                c4b,
                s_int,
                x_int,
                offsets_b,
                offsets,
                Phi_b_b,
                Phi_f_b,
                tp_j,
                rhoL_j,
                True,
                end,
                rk,
                code_bwd,
                own_bwd,
                lin_own_seg,
                lin_ctr_seg,
                phase_end,
            )

            for qq in range(n_flag):
                A_bwd_i[flag_idx[qq]] = own_bwd[flag_idx[qq]]
            if n_flag > 0:
                S_p, S_m = _edge_sources_segmented(
                    xs_edge,
                    sub_off,
                    flag_idx,
                    is_flag,
                    kappas_seg,
                    r3,
                    r4,
                    off3,
                    off4,
                    g3,
                    g4,
                    c3f,
                    c4f,
                    c3b,
                    c4b,
                    x_int,
                    s_int,
                    offsets,
                    Phi_f_b,
                    Phi_b_b,
                    A_fwd_i,
                    A_bwd_i,
                    end,
                    phase_end,
                )
                for qq in range(n_flag):
                    f = flag_idx[qq]
                    pe = phase_end[f]
                    c_p, c_m = _edge_bvp_solve_segmented(
                        np.ascontiguousarray(edge_K_table[i, :, f]),
                        xs_edge,
                        sub_off,
                        np.ascontiguousarray(S_p[qq]),
                        np.ascontiguousarray(S_m[qq]),
                        t_in[f] * y0_array_fwd[i, f],
                        rho_in[f],
                        -t_out[f] * y0_array_bwd[i, f] * pe,
                        rho_out[f] * pe * pe,
                        np.ascontiguousarray(t_j[:, f]),
                        np.ascontiguousarray(tp_j[:, f]),
                        np.ascontiguousarray(rhoL_j[:, f]),
                        np.ascontiguousarray(rhoR_j[:, f]),
                        np.ascontiguousarray(Phi_f_b[f]),
                    )
                    # Back to the internal grids, segment by segment (left and
                    # right limits at the junctions): c+ on x, B(s) = c-(L-s)/pe.
                    for sg in range(n_seg):
                        lo = sub_off[sg]
                        hi = sub_off[sg + 1]
                        xg = xs_edge[lo:hi]
                        cpr = np.ascontiguousarray(c_p[lo:hi].real)
                        cpi = np.ascontiguousarray(c_p[lo:hi].imag)
                        cmr = np.ascontiguousarray(c_m[lo:hi].real)
                        cmi = np.ascontiguousarray(c_m[lo:hi].imag)
                        for t in range(offsets[sg], offsets[sg + 1]):
                            xv = x_int[t]
                            A_fwd_i[f, t] = np.interp(xv, xg, cpr) + 1j * np.interp(
                                xv, xg, cpi
                            )
                        for t in range(n_int - offsets[sg + 1], n_int - offsets[sg]):
                            xm = end - s_int[t]
                            A_bwd_i[f, t] = (
                                np.interp(xm, xg, cmr) + 1j * np.interp(xm, xg, cmi)
                            ) / pe

            # Map to the output grid for storage.
            for j in range(n_modes):
                for kx in range(n_x):
                    A_fwd_x[j, kx] = A_fwd_i[j, take[kx]]
                    A_bwd_x[j, kx] = A_bwd_i[j, take_b[kx]]

            f_k = A_bwd_i - u_k
            scales = _mode_scales(A_bwd_i, scale_floor) + _mode_scales(A_fwd_i, 0.0)
            res = _relative_residual(f_k, scales)
            residuals[i] = res
            # A blown-up (NaN/inf) iterate never recovers: stop without
            # flagging convergence, so the caller reports the point.
            blown_up = not np.isfinite(res)
            if not blown_up and k > 0 and res < convergence_threshold:
                converged = True
            if not converged and not blown_up:
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
            I_out[i + id_add_fwd] = A_fwd_x
            I_out[i + id_add_bwd] = A_bwd_x

            u_k = np.ascontiguousarray(u_next)

            if converged or blown_up:
                converged_flags[i] = converged
                if save_all_passes:
                    for kk in range(k + 1, passes):
                        I_out[i + kk * n_freq] = A_fwd_x
                        I_out[i + kk * n_freq + n_freq * passes] = A_bwd_x
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
    passes: int = _DEFAULTS.passes,
    kappa_cutoff: float = _DEFAULTS.kappa_cutoff,
    m_max: int | np.ndarray = int(_DEFAULTS.m_max),
    convergence_threshold: float = _DEFAULTS.convergence_threshold,
    save_all_passes: bool = _DEFAULTS.save_all_passes,
    anderson_depth: int = _DEFAULTS.anderson_depth,
    process_weights_3wm: FloatArray | None = None,
    process_weights_4wm: FloatArray | None = None,
    Ze: float | ComplexArray = 50.0,
    Zout_in: ComplexArray | None = None,
    Zout_out: ComplexArray | None = None,
    abcd_in: ComplexArray | None = None,
    abcd_out: ComplexArray | None = None,
    rk_config: RKSolverConfig | None = None,
    edge_K: ComplexArray | None = None,
    edge_mode: np.ndarray | None = None,
    diagnostics: dict | None = None,
) -> dict[str, np.ndarray]:
    """
    Solve the forward/backward Bloch CMEs of a segmented device.

    Parameters
    ----------
    x_array : (n_x,) float
        Output grid on ``[0, L]``; must be symmetric under ``x -> L - x``
        (e.g. ``linspace(0, L, n)``).
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
        Real factors of every relation row per segment; default 1.
    edge_K : (n_freq, n_seg, n_modes, 2, 2) complex, optional
        Per-unit-cell linear generator of ``(c+, c-)`` of every edge mode in
        every segment: the fixed-basis ``K`` where the mode is flagged,
        ``diag(-alpha, alpha)`` in its native segments.
    edge_mode : (n_freq, n_modes) bool, optional
        Modes solved exactly across the line (flagged in at least one
        segment). Their tables, ``prefac`` and kernel gammas must already be
        the edge-basis ones where flagged. Default: none.
    rk_config : RKSolverConfig, optional
        Settings of the RK45 integrator (default ``RKSolverConfig()``).
    m_max : int or (n_freq, n_seg, n_modes) int array
        Harmonic limit, global or per frequency, segment and mode (a process
        uses the largest limit among its modes).
    passes, kappa_cutoff, convergence_threshold, save_all_passes, anderson_depth
        As in :class:`BlochSolverConfig`, whose defaults (and ``m_max``) they take.
    diagnostics : dict, optional
        If given, filled with ``converged`` and ``residuals`` (n_freq,).

    Returns
    -------
    dict
        ``I_fwd``, ``I_bwd`` : (n_saved, n_freq, n_modes, n_x)
            Forward and backward mode currents (with the propagation phase) on
            ``x_array``; ``n_saved`` is ``passes`` with ``save_all_passes``, else 1.
        ``a_in``, ``a_out`` : (n_freq, n_modes)
            Currents incident from the external lines at x=0 and x=L.
        ``b_in``, ``b_out`` : (n_saved, n_freq, n_modes)
            Currents going out into the external lines at x=0 and x=L: the
            transmitted device wave plus the prompt reflection of the incident one.
    """
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
    if anderson_depth < 1:
        raise ValueError(f"anderson_depth must be >= 1, got {anderson_depth}.")

    edge_mode_f: np.ndarray
    if edge_mode is None:
        edge_mode_f = np.zeros((n_freq, n_modes), dtype=np.float64)
    else:
        edge_mode_f = np.asarray(edge_mode, dtype=np.float64)
    if edge_K is None:
        edge_K = np.zeros((n_freq, n_seg, n_modes, 2, 2), dtype=np.complex128)

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
        c(np.broadcast_to(np.asarray(m_max, dtype=np.int64), (n_freq, n_seg, n_modes))),
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
        (rk_config if rk_config is not None else RKSolverConfig()).kernel_args(),
        c(edge_K, dtype=np.complex128),
        c(edge_mode_f),
    )
    if not np.all(converged):
        bad = np.flatnonzero(~converged)
        log.warning(
            "Bloch forward-backward CMEs did not converge for %d of %d "
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
    if diagnostics is not None:
        diagnostics["converged"] = converged
        diagnostics["residuals"] = residuals

    n_saved = passes if save_all_passes else 1
    raw = I_out.reshape((2, n_saved, n_freq, n_modes, len(x_array)))
    Phi_f, Phi_b = segmented_phase(
        np.transpose(gammas_seg.imag, (0, 2, 1)), seg_bounds, x_array
    )
    I_fwd = raw[0] * np.exp(1j * Phi_f)
    # Backward envelopes are solved on s = L - x; x_array is symmetric.
    I_bwd = (raw[1] * np.exp(1j * Phi_b))[..., ::-1]

    a_in = np.asarray(y0_array_fwd, dtype=np.complex128)
    a_out = np.asarray(y0_array_bwd, dtype=np.complex128)
    Zp0, Zm0, ZpL, ZmL = Zp[:, 0], Zm[:, 0], Zp[:, -1], Zm[:, -1]
    rho_in = (Zm0 - Zout_in_table) / (Zp0 + Zout_in_table)
    rho_out = (ZpL - Zout_out_table) / (ZmL + Zout_out_table)
    b_in = (
        tof_in * (1 + rho_in) * I_bwd[..., 0]
        + _prompt_reflection(abcd_in, Zp0, Ze_table) * a_in
    )
    b_out = (
        -tof_out * (1 + rho_out) * I_fwd[..., -1]
        + _prompt_reflection(abcd_out, ZmL, Ze_table) * a_out
    )
    return {
        "I_fwd": I_fwd,
        "I_bwd": I_bwd,
        "a_in": a_in,
        "a_out": a_out,
        "b_in": b_in,
        "b_out": b_out,
    }


def _prompt_reflection(abcd, Zb_launch, Ze):
    """
    Current reflected into the external line per unit incident current, off the
    connector ``abcd`` (port 2 on the device) loaded by the launched Bloch mode.
    """
    if abcd is None:
        Zin = Zb_launch
    else:
        A, B, C, D = (np.asarray(abcd)[..., k] for k in range(4))
        Zin = (A * Zb_launch + B) / (C * Zb_launch + D)
    return (Ze - Zin) / (Ze + Zin)
