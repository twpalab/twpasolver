"""
Linear Bloch-mode tables for the coupled mode equations in the Bloch basis.

The standard CMEs treat the periodically loaded line as a homogeneous medium:
the supercell only provides the Bloch propagation constant, while the
nonlinear coupling of mode ``n`` is scaled by its propagation constant
(``-gamma_n`` in the kernel). In the Bloch basis the amplitudes are the
boundary currents of the forward Bloch modes, and the coupling becomes

    dA_n/dx  ∋  -(2j omega_n / W_n) * c_q * <L_j phi^-_{n,j} prod_r phi_{p_r,j}>,

where

* ``W_n = Zb^+_n + Zb^-_n`` is the reciprocity invariant <psi^+, psi^->
  (the mode power normalization, ``2 Re Zb`` for a lossless line),
* ``phi^±_{n,j}`` is the periodic part of the current of the forward/backward
  Bloch mode through the series nonlinear inductor ``j`` of the supercell,
  normalized to unit current at the supercell boundary,
* ``<...>`` is the sum over the nonlinear inductors divided by the number of
  unit cells per supercell (``x`` is measured in unit cells).

For a homogeneous lumped ladder ``phi = 1`` and ``-2j omega L / W`` equals
``-gamma * tanh(gamma/2) / (gamma/2)``, which tends to the standard ``-gamma``
far below the ladder cutoff.

This module provides the linear ingredients: Bloch impedances, ``W`` and the
intracell current profiles of a supercell (:func:`bloch_mode_tables`), the
fixed ("edge") basis for modes close to a stopband edge
(:func:`edge_basis_tables` and the edge search), and the per-process weights
of the stopband guard (:func:`process_weights`). The forward-backward solver is in
:mod:`twpasolver.cmes.bloch_fb`. Supported unit cells are those whose
nonlinearity is a lumped series inductance (``LCLfBaseCell``,
``StubBaseCell``), or any cell defining a ``bloch_nl_elements()`` method (see
:func:`series_nl_elements`).

Time convention is the package one: exp(+j omega t), forward waves ~ exp(-gamma x)
with gamma = alpha + j k.
"""

from __future__ import annotations

import math
from typing import Any

import numba as nb
import numpy as np

from twpasolver.bonus_types import ComplexArray, FloatArray
from twpasolver.logger import log
from twpasolver.models.twoportarrays import TwoPortArray
from twpasolver.models.twpa_cells import LCLfBaseCell, StubBaseCell

# Unit-cell port whose current flows through a series nonlinear inductor.
PORT_IN = 0
PORT_OUT = 1


# --------------------------------------------------------------------------- #
# Linear part: unit cells, Bloch modes and intracell current profiles
# --------------------------------------------------------------------------- #


def series_nl_elements(cell: Any) -> list[tuple[int, float]]:
    """
    Return the series nonlinear inductors of a unit cell.

    Each element is ``(port, L)``: the nonlinear inductance ``L`` carries the
    current of the cell's input (``PORT_IN``) or output (``PORT_OUT``) port.
    Custom cells can define a ``bloch_nl_elements()`` method returning the same
    list to be supported. The package cells scale ``L`` by their ``nl_scale``.
    """
    hook = getattr(cell, "bloch_nl_elements", None)
    if callable(hook):
        return list(hook())
    if isinstance(cell, LCLfBaseCell):
        # Shunt-first cell [Y][L]; centered variant is [L/2][Y][L/2].
        L = cell.L * cell.nl_scale
        if cell.centered:
            return [(PORT_IN, L / 2), (PORT_OUT, L / 2)]
        return [(PORT_OUT, L)]
    if isinstance(cell, StubBaseCell):
        L = cell.L * cell.nl_scale
        if cell.line:
            # Lumped centre line split around the stubs: [L/2][stubs][L/2].
            return [(PORT_IN, L / 2), (PORT_OUT, L / 2)]
        # get_stub_cell is shunt-first [Y][L*l2].
        return [(PORT_OUT, L * cell.l2)]
    raise NotImplementedError(
        f"Bloch CMEs: no nonlinear-element description for {type(cell).__name__}. "
        "Define a bloch_nl_elements() method returning [(port, L), ...]."
    )


def expand_unit_cells(array: TwoPortArray) -> list[tuple[Any, int]]:
    """
    Expand one period (supercell) of ``array`` into ordered runs ``(cell, count)``.

    The top-level repetition ``array.N`` is not applied (it is the number of
    supercells); nested ``TwoPortArray`` instances are expanded recursively.
    """
    runs: list[tuple[Any, int]] = []
    for cell in array.cells:
        if isinstance(cell, TwoPortArray):
            runs.extend(expand_unit_cells(cell) * int(cell.N))
        else:
            runs.append((cell, int(cell.N)))
    return [(cell, n) for cell, n in runs if n > 0]


def _eigvec_ratio(T: np.ndarray, lam: np.ndarray) -> np.ndarray:
    """V/I of the ABCD eigenvector with eigenvalue ``lam``, best-conditioned form."""
    A, B, C, D = T[:, 0, 0], T[:, 0, 1], T[:, 1, 0], T[:, 1, 1]
    den1 = lam - A
    use_first = np.abs(den1) >= np.abs(C)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(use_first, B / den1, (lam - D) / C)


def _inverse_2x2(T: np.ndarray) -> np.ndarray:
    det = T[:, 0, 0] * T[:, 1, 1] - T[:, 0, 1] * T[:, 1, 0]
    Tinv = np.empty_like(T)
    Tinv[:, 0, 0] = T[:, 1, 1]
    Tinv[:, 0, 1] = -T[:, 0, 1]
    Tinv[:, 1, 0] = -T[:, 1, 0]
    Tinv[:, 1, 1] = T[:, 0, 0]
    return Tinv / det[:, None, None]


def _t_split(Tc: np.ndarray) -> list[tuple[np.ndarray, ...]]:
    """
    ``(A, B, C, D)`` of the series-shunt-series (T) network equal to ``Tc``.

    Exact for any reciprocal two-port (det = 1); for the package lumped cells
    these are their actual elements. Where ``C == 0`` (a pure series element)
    the middle step is ``Tc`` itself.
    """
    A, B, C, D = Tc[:, 0, 0], Tc[:, 0, 1], Tc[:, 1, 0], Tc[:, 1, 1]
    split = C != 0
    one, zero = np.ones_like(A), np.zeros_like(A)
    with np.errstate(divide="ignore", invalid="ignore"):
        Z1 = np.where(split, (A - 1) / C, 0)
        Z2 = np.where(split, (D - 1) / C, 0)
    mid = (np.where(split, 1, A), np.where(split, 0, B), C, np.where(split, 1, D))
    return [(one, Z1, zero, one), mid, (one, Z2, zero, one)]


@nb.njit(parallel=True, cache=True)
def _wind_backward(steps, counts, V, I, Z0):
    """
    Phase of ``a = V + Z0 I`` accumulated from x = d back to x = 0.

    ``steps[r, s]`` are the ``(A, B, C, D)`` of step ``s`` of unit-cell run
    ``r``, repeated ``counts[r]`` times. A backward sweep with the plain ABCDs
    is stable: the forward mode grows towards x = 0, so rounding never seeds
    the backward mode. Returns ``(k d, largest single-step phase)``.
    """
    n_runs, n_steps, _, n_f = steps.shape
    kd = np.zeros(n_f)
    max_step = np.zeros(n_f)
    for f in nb.prange(n_f):
        a = V[f] + Z0 * I[f]
        v, i = V[f] / a, I[f] / a  # normalised so that a = 1 before each step
        acc = 0.0
        big = 0.0
        for r in range(n_runs - 1, -1, -1):
            for _ in range(counts[r]):
                for s in range(n_steps - 1, -1, -1):
                    v, i = (
                        steps[r, s, 0, f] * v + steps[r, s, 1, f] * i,
                        steps[r, s, 2, f] * v + steps[r, s, 3, f] * i,
                    )
                    a = v + Z0 * i
                    step = math.atan2(a.imag, a.real)
                    acc += step
                    big = max(big, abs(step))
                    v, i = v / a, i / a
        kd[f] = acc
        max_step[f] = big
    return kd, max_step


def bloch_wavenumber(
    twpa: TwoPortArray, freqs_hz: FloatArray, T: np.ndarray, Z0: complex
) -> np.ndarray:
    """
    Bloch wavenumber ``k d`` per supercell of the forward mode, by winding.

    The forward mode is the eigenvector of the supercell ABCD ``T`` carrying
    power along +x (``Re Zb > 0``) in a passband and decaying along +x
    (``|lambda| > 1``) in a stopband. Its forward-wave amplitude
    ``a = V + Z0 I`` is carried from x = d back to x = 0 element by element
    (series/shunt steps of every unit cell) and ``k d`` is the accumulated phase
    of ``a``. Unlike unwrapping ``arg lambda`` along the frequency grid this does
    not depend on the grid, gives ``k`` near 0 above a notch and ``k < 0`` in a
    left-handed band. The result is snapped to ``arg lambda`` modulo 2 pi.
    """
    freqs_hz = np.atleast_1d(np.asarray(freqs_hz, dtype=np.float64))
    T = np.asarray(T)
    tr = T[:, 0, 0] + T[:, 1, 1]
    root = np.sqrt(tr**2 - 4 + 0j)
    lam1, lam2 = (tr + root) / 2, (tr - root) / 2
    Zb1, Zb2 = _eigvec_ratio(T, lam1), _eigvec_ratio(T, lam2)
    gap = np.abs(np.abs(lam1) - np.abs(lam2)) > 1e-9
    pick1 = np.where(gap, np.abs(lam1) >= np.abs(lam2), Zb1.real > 0)
    lam = np.where(pick1, lam1, lam2)
    V = np.where(pick1, Zb1, Zb2)
    I = np.ones_like(V)

    runs = expand_unit_cells(twpa)
    steps = np.array(
        [
            _t_split(np.asarray(cell.single_abcd(freqs_hz)).astype(np.complex128))
            for cell, _ in runs
        ],
        dtype=np.complex128,
    )  # (n_runs, 3 steps, 4 entries, F)
    counts = np.array([n for _, n in runs], dtype=np.int64)
    kd, max_step = _wind_backward(steps, counts, V, I, complex(Z0))

    arg = np.angle(lam)
    m = np.round((kd - arg) / (2 * np.pi))
    resid = np.abs(kd - arg - 2 * np.pi * m)
    if np.nanmax(resid, initial=0) > 1e-8:
        log.warning(
            "Bloch winding differs from arg(lambda) by up to %.1e rad.",
            np.nanmax(resid),
        )
    if np.nanmax(max_step, initial=0) > 0.95 * np.pi:
        log.warning(
            "Bloch winding: a single element turns the wave phase by %.2f rad; "
            "k may be off by 2 pi there (split the element).",
            np.nanmax(max_step),
        )
    return arg + 2 * np.pi * m


def bloch_mode_tables(
    twpa: TwoPortArray, freqs_hz: FloatArray, gamma_ref: ComplexArray
) -> dict[str, Any]:
    """
    Compute Bloch impedances, W and intracell current profiles of a supercell.

    Parameters
    ----------
    twpa : TwoPortArray
        Non-segmented device; one period is ``twpa.cells``.
    freqs_hz : (F,) float
        Frequencies in Hz (mode frequencies, flattened).
    gamma_ref : (F,) complex
        Per-unit-cell propagation constant ``alpha + j k`` used by the CME
        kernel. It selects the forward eigenvalue and the branch of ``k``.

    Returns
    -------
    dict with
        ``Zb_plus``, ``Zb_minus``, ``W`` : (F,) complex
        ``phi_plus``, ``phi_minus`` : (F, n_nl) complex, periodic profiles of the
            forward/backward mode current through each nonlinear inductor
        ``L_nl`` : (n_nl,) nonlinear inductances
        ``n_cells`` : unit cells per supercell
        ``positions`` : (n_nl,) position of each nonlinear inductor inside the
            supercell, in unit cells (the reference used for ``phi``)
        ``gamma_exact`` : (F,) complex, per-unit-cell gamma from the eigenvalue
        ``eig_residual`` : (F,) float, |I_M * lambda - 1| after propagating the
            forward mode through one supercell (should be ~machine precision)
        ``split`` : (F,) float, ``|sinh(gamma d)| = |lambda+ - lambda-| / 2``,
            the normalisation-free splitting of the two Bloch modes (0 at every
            band edge, the exceptional point of the supercell matrix)
    """
    freqs_hz = np.asarray(freqs_hz, dtype=np.float64)
    gamma_ref = np.asarray(gamma_ref, dtype=np.complex128)
    runs = expand_unit_cells(twpa)
    n_cells = sum(n for _, n in runs)
    n_f = len(freqs_hz)

    T = supercell_matrix(twpa, freqs_hz)

    # Forward mode: psi(d) = psi(0)/lambda with lambda = exp(gamma * n_cells).
    tr = T[:, 0, 0] + T[:, 1, 1]
    root = np.sqrt(tr**2 - 4 + 0j)
    lam1 = (tr + root) / 2
    lam2 = (tr - root) / 2
    target = np.exp(gamma_ref * n_cells)
    pick1 = np.abs(lam1 - target) <= np.abs(lam2 - target)
    lam_f = np.where(pick1, lam1, lam2)
    lam_b = np.where(pick1, lam2, lam1)

    Zb_plus = _eigvec_ratio(T, lam_f)  # psi+ = ( Zb+, 1)
    Zb_minus = -_eigvec_ratio(T, lam_b)  # psi- = (-Zb-, 1)
    W = Zb_plus + Zb_minus

    g0 = np.log(lam_f) / n_cells
    shift = np.round((gamma_ref.imag - g0.imag) * n_cells / (2 * np.pi))
    gamma_exact = g0 + 2j * np.pi * shift / n_cells

    psi_p = np.stack([Zb_plus, np.ones(n_f, dtype=np.complex128)], axis=-1)
    psi_m = np.stack([-Zb_minus, np.ones(n_f, dtype=np.complex128)], axis=-1)
    J_p, J_m, L_nl, planes, I_end = _intracell_currents(twpa, freqs_hz, psi_p, psi_m)
    pos = planes[None, :]
    phi_plus = J_p * np.exp(gamma_exact[:, None] * pos)
    phi_minus = J_m * np.exp(-gamma_exact[:, None] * pos)

    return {
        "Zb_plus": Zb_plus,
        "Zb_minus": Zb_minus,
        "W": W,
        "phi_plus": phi_plus,
        "phi_minus": phi_minus,
        "L_nl": L_nl,
        "n_cells": n_cells,
        "positions": planes,
        "gamma_exact": gamma_exact,
        "eig_residual": np.abs(I_end * lam_f - 1),
        "split": np.abs(lam_f - lam_b) / 2,
    }


# --------------------------------------------------------------------------- #
# Fixed ("edge") Bloch basis for modes close to a stopband edge
# --------------------------------------------------------------------------- #
#
# At a band edge the supercell matrix T is a Jordan block: psi+ and psi-
# coalesce, W -> 0 and every projected coupling grows as 1/W. A flagged mode is
# instead expanded on a *fixed* pair psi~+- taken from the lossless supercell
# at a reference frequency f_ref on the passband side of the edge. Any fixed
# invertible basis is exact; one supercell then maps the amplitudes with the
# non-diagonal matrix P = U^-1 T(f)^-1 U, whose off-diagonal part is the Bragg
# coupling between the two directions. With the edge phase factored out,
# c+- = a~+- exp(+-j kappa_e x), kappa_e d = m pi, the amplitudes obey
#     dc/dx = K c + (nonlinear source),   K = log((-1)^m P) / d,
# a constant 2x2 matrix that is regular at the exceptional point (it is never
# diagonalised).


def lossless_copy(model: Any) -> Any:
    """Deep copy of a device model with every loss tangent ``delta`` set to zero."""
    twin = model.model_copy(deep=True)

    def _strip(obj):
        if hasattr(obj, "delta"):
            object.__setattr__(obj, "delta", 0.0)
        for cell in getattr(obj, "cells", None) or []:
            _strip(cell)

    _strip(twin)
    return twin


def supercell_matrix(twpa: TwoPortArray, freqs_hz: FloatArray) -> np.ndarray:
    """ABCD matrix of one supercell, shape (F, 2, 2)."""
    freqs_hz = np.atleast_1d(np.asarray(freqs_hz, dtype=np.float64))
    T = None
    for cell, n in expand_unit_cells(twpa):
        with np.errstate(over="ignore", invalid="ignore"):
            Tc = np.linalg.matrix_power(
                np.asarray(cell.single_abcd(freqs_hz), dtype=np.complex128), n
            )
            T = Tc if T is None else T @ Tc
    if T is None:
        return np.broadcast_to(
            np.eye(2, dtype=np.complex128), (len(freqs_hz), 2, 2)
        ).copy()
    return T


def logm_unimodular_2x2(Q: np.ndarray) -> np.ndarray:
    """
    Principal logarithm of 2x2 matrices with unit determinant, shape (..., 2, 2).

    ``log Q = (phi / sinh phi) (Q - cosh(phi) I)`` with ``cosh phi = tr(Q)/2``,
    regular at the exceptional point ``Q - I`` nilpotent (``log Q = Q - I``).
    Valid for Q close to the identity (|phi| well below pi).
    """
    Q = np.asarray(Q, dtype=np.complex128)
    c = 0.5 * (Q[..., 0, 0] + Q[..., 1, 1])
    phi = np.arccosh(c)
    small = np.abs(phi) < 1e-4
    safe = np.where(small, 1.0, phi)
    ratio = np.where(small, 1.0 + phi**2 / 6.0, safe / np.sinh(safe))
    eye = np.eye(2, dtype=np.complex128)
    return ratio[..., None, None] * (Q - c[..., None, None] * eye)


# |tr T / 2| must exceed 1 by more than this for a frequency to count as inside
# a stopband. At the zone-folding points of a supercell made of repeated
# periods (e.g. a 34-cell holotype unfolded to 102 cells) the bands touch
# without a gap, |tr T / 2| = 1 up to rounding, and the splitting
# |sinh gamma d| vanishes there although the Bloch modes stay distinct.
STOPBAND_TOL = 1e-6


def passband_mask(h: np.ndarray) -> np.ndarray:
    """
    Passband points of a scan of the half trace ``h = tr T / 2``.

    ``|h| < 1``, except that runs of points with ``|h| >= 1`` whose excess never
    reaches ``STOPBAND_TOL`` (bands touching at a folding point, rounding) are
    counted as passband.
    """
    a = np.abs(np.asarray(h).real)
    inband = a < 1.0
    out = np.flatnonzero(~inband)
    if out.size:
        runs = np.split(out, np.flatnonzero(np.diff(out) > 1) + 1)
        for r in runs:
            if a[r].max() - 1.0 <= STOPBAND_TOL:
                inband[r] = True
    return inband


def in_stopband(twpa_ll: TwoPortArray, f_hz: FloatArray) -> np.ndarray:
    """True where ``|tr T / 2| > 1 + STOPBAND_TOL`` for the (lossless) supercell."""
    T = supercell_matrix(twpa_ll, np.atleast_1d(np.asarray(f_hz, dtype=np.float64)))
    return np.abs(0.5 * (T[:, 0, 0] + T[:, 1, 1]).real) > 1.0 + STOPBAND_TOL


def _half_trace(twpa_ll: TwoPortArray, f) -> np.ndarray:
    T = supercell_matrix(twpa_ll, np.atleast_1d(f))
    return 0.5 * (T[:, 0, 0] + T[:, 1, 1]).real


def _refine_edge(
    twpa_ll: TwoPortArray,
    fs: np.ndarray,
    h: np.ndarray,
    inband: np.ndarray,
    j: int,
    h_ref: float,
) -> tuple[float, float, int]:
    """
    Edge between ``fs[j]`` and ``fs[j + 1]`` of a scan ``h`` of ``tr T / 2``.

    Returns the refined edge, the reference frequency on its passband side
    where ``|h| = h_ref`` (NaN if the scanned passband does not reach it) and
    the parity ``sign(h)`` at the edge.
    """
    from scipy.optimize import brentq

    f_edge = brentq(
        lambda f: abs(_half_trace(twpa_ll, f)[0]) - 1.0, fs[j], fs[j + 1], xtol=1.0
    )
    parity = int(np.sign(h[j] if not inband[j] else h[j + 1]))
    # Walk from the edge into the passband until |h| < h_ref.
    step = 1 if inband[j + 1] else -1
    k = j + 1 if step == 1 else j
    while 0 <= k < len(fs) and inband[k] and abs(h[k]) >= h_ref:
        k += step
    f_ref = np.nan
    if 0 <= k < len(fs) and inband[k]:
        # |h| - h_ref changes sign between fs[k - step] (outside the band or
        # above h_ref) and fs[k].
        a, b = sorted((fs[k - step], fs[k]))
        f_ref = brentq(
            lambda f: abs(_half_trace(twpa_ll, f)[0]) - h_ref, a, b, xtol=1.0
        )
    return f_edge, f_ref, parity


def edge_reference_frequency(
    twpa_ll: TwoPortArray,
    f_hz: float,
    e_ref: float,
    span_hz: float = 0.5e9,
    n_scan: int = 2001,
    max_span_hz: float = 8e9,
) -> tuple[float, float, int]:
    """
    Reference frequency of the fixed basis of a mode near a stopband edge.

    Finds the band edge of the lossless supercell closest to ``f_hz``
    (``|tr T / 2| = 1``) and, on its passband side, the frequency where the
    splitting ``|sinh(gamma d)| = sqrt(1 - (tr T / 2)^2)`` equals ``e_ref``.

    Returns
    -------
    f_ref, f_edge : float
        Reference and edge frequencies (Hz).
    parity : int
        ``(-1)^m = sign(tr T / 2)`` at the edge.
    """
    h_ref = np.sqrt(1.0 - e_ref**2)
    span = span_hz
    while span <= max_span_hz:
        fs = f_hz + np.linspace(-span, span, n_scan)
        fs = fs[fs > 0]
        h = _half_trace(twpa_ll, fs)
        inband = passband_mask(h)
        cross = np.flatnonzero(inband[1:] != inband[:-1])
        if cross.size:
            j = cross[np.argmin(np.abs(fs[cross] - f_hz))]
            f_edge, f_ref, parity = _refine_edge(twpa_ll, fs, h, inband, j, h_ref)
            if not np.isnan(f_ref):
                return float(f_ref), float(f_edge), parity
        span *= 2
    raise ValueError(
        f"No stopband edge with a passband reaching split {e_ref} found around "
        f"{f_hz / 1e9:.4f} GHz."
    )


def stopband_edges(
    twpa_ll: TwoPortArray,
    f_hz: FloatArray,
    e_ref: float,
    span_hz: float = 1.0e9,
    step_hz: float = 0.5e6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Stopband edges of a lossless supercell near a set of frequencies.

    Scans ``|tr T / 2|`` on a grid of step ``step_hz`` over the union of the
    intervals ``f +- span_hz`` and refines every edge, ``|tr T / 2| = 1``, and
    its reference frequency on the passband side, where the splitting equals
    ``e_ref``, as :func:`edge_reference_frequency` does for a single frequency.

    Returns
    -------
    f_edge, f_ref : (E,) float
        Edge and reference frequencies (Hz), sorted by edge; ``f_ref`` is NaN
        when the passband next to the edge does not reach ``e_ref`` within the
        scanned block.
    parity : (E,) int
        ``(-1)^m = sign(tr T / 2)`` at the edge.
    """
    h_ref = np.sqrt(1.0 - e_ref**2)
    n_span = int(np.ceil(span_hz / step_hz))
    centres = np.unique(np.round(np.asarray(f_hz, dtype=np.float64) / step_hz)).astype(
        np.int64
    )
    edges, refs, pars = [], [], []
    if centres.size == 0:
        return np.empty(0), np.empty(0), np.empty(0, dtype=np.int64)
    # merge the intervals [c - n_span, c + n_span] into contiguous blocks
    blocks = []
    lo, hi = centres[0] - n_span, centres[0] + n_span
    for c in centres[1:]:
        if c - n_span <= hi + 1:
            hi = c + n_span
        else:
            blocks.append((lo, hi))
            lo, hi = c - n_span, c + n_span
    blocks.append((lo, hi))
    for lo, hi in blocks:
        g = np.arange(max(lo, 1), hi + 1) * step_hz
        if len(g) < 2:
            continue
        h = _half_trace(twpa_ll, g)
        inband = passband_mask(h)
        for j in np.flatnonzero(inband[1:] != inband[:-1]):
            f_edge, f_ref, parity = _refine_edge(twpa_ll, g, h, inband, j, h_ref)
            edges.append(f_edge)
            refs.append(f_ref)
            pars.append(parity)
    order = np.argsort(edges)
    return (
        np.asarray(edges)[order],
        np.asarray(refs, dtype=np.float64)[order],
        np.asarray(pars, dtype=np.int64)[order],
    )


def _intracell_currents(
    twpa: TwoPortArray, freqs_hz: FloatArray, psi_p: np.ndarray, psi_m: np.ndarray
):
    """Currents of two boundary vectors propagated through one supercell.

    Returns the currents through every nonlinear inductor ``(F, n_nl)`` for
    each vector, the inductances, the positions (unit cells) and the current
    of the first vector at the end of the supercell ``(F,)``.
    """
    runs = expand_unit_cells(twpa)
    J_p, J_m, L_nl, planes = [], [], [], []
    plane = 0
    for cell, n in runs:
        Tc = np.asarray(cell.single_abcd(freqs_hz)).astype(np.complex128)
        elements = series_nl_elements(cell)
        Tinv = _inverse_2x2(Tc)
        for _ in range(n):
            out_p = np.einsum("fij,fj->fi", Tinv, psi_p)
            out_m = np.einsum("fij,fj->fi", Tinv, psi_m)
            for port, L in elements:
                J_p.append((psi_p if port == PORT_IN else out_p)[:, 1])
                J_m.append((psi_m if port == PORT_IN else out_m)[:, 1])
                L_nl.append(L)
                planes.append(plane + port)
            psi_p, psi_m = out_p, out_m
            plane += 1
    return (
        np.stack(J_p, axis=-1),
        np.stack(J_m, axis=-1),
        np.asarray(L_nl, dtype=np.float64),
        np.asarray(planes, dtype=np.float64),
        psi_p[:, 1],
    )


def edge_basis_tables(
    twpa: TwoPortArray,
    twpa_ll: TwoPortArray,
    f_hz: FloatArray,
    f_ref_hz: FloatArray,
    m_edge: np.ndarray,
) -> dict[str, Any]:
    """
    Fixed-basis tables of modes close to a stopband edge.

    Parameters
    ----------
    twpa, twpa_ll : TwoPortArray
        Lossy device and its lossless copy (same supercell).
    f_hz : (F,) float
        Actual mode frequencies.
    f_ref_hz : (F,) float
        Reference frequencies (passband side, see
        :func:`edge_reference_frequency`).
    m_edge : (F,) int
        Edge index, ``kappa_e d = m pi``.

    Returns
    -------
    dict with
        ``Zb_plus``, ``Zb_minus`` : (F,) lossless Bloch impedances at ``f_ref``
            (the fixed boundary vectors ``(Zb+, 1)``, ``(-Zb-, 1)``)
        ``W`` : (F,) real, ``Zb+ + Zb-`` (positive)
        ``phi_plus``, ``phi_minus`` : (F, n_nl) periodic profiles of the fixed
            vectors propagated through the **lossy** cells at ``f``, with the
            edge phase ``exp(+-j kappa_e pos)``
        ``K``, ``K_lossless`` : (F, 2, 2) per-unit-cell linear generator of
            ``(c+, c-)`` from the lossy / lossless supercell at ``f``
        ``kappa_e`` : (F,) edge wavenumber per unit cell
        ``ham_defect`` : (F,) ``||K^dag S + S K|| / ||K||`` of the lossless K,
            ``S = diag(1, -1)``; zero to rounding (flux conservation)
    """
    f_hz = np.atleast_1d(np.asarray(f_hz, dtype=np.float64))
    f_ref_hz = np.atleast_1d(np.asarray(f_ref_hz, dtype=np.float64))
    m_edge = np.atleast_1d(np.asarray(m_edge, dtype=np.int64))
    n_cells = sum(n for _, n in expand_unit_cells(twpa))
    n_f = len(f_hz)

    T_ref = supercell_matrix(twpa_ll, f_ref_hz)
    tr = T_ref[:, 0, 0] + T_ref[:, 1, 1]
    root = np.sqrt(tr**2 - 4 + 0j)
    lam1, lam2 = (tr + root) / 2, (tr - root) / 2
    r1, r2 = _eigvec_ratio(T_ref, lam1), _eigvec_ratio(T_ref, lam2)
    # Forward vector: the one carrying positive power, Re W > 0.
    fwd1 = (r1 - r2).real > 0
    Zb_plus = np.where(fwd1, r1, r2)
    Zb_minus = -np.where(fwd1, r2, r1)
    W = (Zb_plus + Zb_minus).real

    U = np.zeros((n_f, 2, 2), dtype=np.complex128)
    U[:, 0, 0], U[:, 0, 1] = Zb_plus, -Zb_minus
    U[:, 1, :] = 1.0
    Uinv = np.linalg.inv(U)
    parity = np.where(m_edge % 2 == 0, 1.0, -1.0)[:, None, None]
    S = np.diag([1.0, -1.0])
    out_K = {}
    for name, model in (("K", twpa), ("K_lossless", twpa_ll)):
        P = Uinv @ _inverse_2x2(supercell_matrix(model, f_hz)) @ U
        out_K[name] = logm_unimodular_2x2(parity * P) / n_cells
    Kl = out_K["K_lossless"]
    ham = np.linalg.norm(
        np.conj(np.swapaxes(Kl, -1, -2)) @ S + S @ Kl, axis=(-2, -1)
    ) / np.maximum(np.linalg.norm(Kl, axis=(-2, -1)), 1e-300)

    psi_p = np.stack([Zb_plus, np.ones(n_f, dtype=np.complex128)], axis=-1)
    psi_m = np.stack([-Zb_minus, np.ones(n_f, dtype=np.complex128)], axis=-1)
    J_p, J_m, L_nl, pos, _ = _intracell_currents(twpa, f_hz, psi_p, psi_m)
    kappa_e = m_edge * np.pi / n_cells
    phase = np.exp(1j * kappa_e[:, None] * pos[None, :])
    return {
        "Zb_plus": Zb_plus,
        "Zb_minus": Zb_minus,
        "W": W,
        "phi_plus": J_p * phase,
        "phi_minus": J_m / phase,
        "K": out_K["K"],
        "K_lossless": Kl,
        "kappa_e": kappa_e,
        "ham_defect": ham,
        "n_cells": n_cells,
    }


def reactive_angle_deg(W: np.ndarray) -> np.ndarray:
    """
    Reactive angle of the Bloch normalisation, ``arcsin(|Im W| / |W|)`` in degrees.

    0 for a power-carrying travelling Bloch mode (``W`` real in every lossless
    passband, for any supercell reference plane), 90 for a purely evanescent
    mode inside a lossless stopband. Unlike ``arg Zb`` it does not depend on the
    supercell boundary chosen for asymmetric cells.
    """
    W = np.asarray(W)
    ratio = np.abs(W.imag) / np.maximum(np.abs(W), 1e-300)
    return np.degrees(np.arcsin(np.clip(ratio, 0.0, 1.0)))


def process_weights(relations: np.ndarray, mode_weights: np.ndarray) -> np.ndarray:
    """
    Weight of every relation row: product of the weights of all its modes.

    All rows describing the same physical process (one per participating mode,
    and every direction / reciprocal-vector variant in the FB model) contain the
    same multiset of modes, so they receive the same real factor and the
    Manley-Rowe cancellation between them is preserved exactly.

    Parameters
    ----------
    relations : (R, 1+p) int
        Rows ``[target, input_1, ...]``; inputs ``>= n_modes`` are conjugated.
    mode_weights : (..., n_modes) float

    Returns
    -------
    (..., R) float
    """
    n_modes = mode_weights.shape[-1]
    out = np.ones(mode_weights.shape[:-1] + (len(relations),), dtype=np.float64)
    for r, rel in enumerate(relations):
        for idx in rel:
            out[..., r] *= mode_weights[..., int(idx) % n_modes]
    return out
