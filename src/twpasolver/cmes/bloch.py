"""
Coupled mode equations in the Bloch-mode basis.

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

Scope of this first version:
    * no-reflections model (forward modes only),
    * non-segmented devices,
    * unit cells whose nonlinearity is a lumped series inductance
      (``LCLfBaseCell``, ``StubBaseCell``), or any cell defining a
      ``bloch_nl_elements()`` method (see :func:`series_nl_elements`).

Time convention is the package one: exp(+j omega t), forward waves ~ exp(-gamma x)
with gamma = alpha + j k.
"""

from __future__ import annotations

from typing import Any

import numba as nb
import numpy as np

from twpasolver.bonus_types import (
    ComplexArray,
    FloatArray,
    nb_complex1d,
    nb_float1d,
    nb_int2d,
)
from twpasolver.cmes.rk45 import rk45_solve
from twpasolver.cmes.solver_config import (
    SOLVER_ATOL,
    SOLVER_FIRST_STEP,
    SOLVER_MAX_STEPS,
    SOLVER_RTOL,
)
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
    list to be supported.
    """
    hook = getattr(cell, "bloch_nl_elements", None)
    if callable(hook):
        return list(hook())
    if isinstance(cell, LCLfBaseCell):
        # Shunt-first cell [Y][L]; centered variant is [L/2][Y][L/2].
        if cell.centered:
            return [(PORT_IN, cell.L / 2), (PORT_OUT, cell.L / 2)]
        return [(PORT_OUT, cell.L)]
    if isinstance(cell, StubBaseCell):
        if cell.line:
            # Lumped centre line split around the stubs: [L/2][stubs][L/2].
            return [(PORT_IN, cell.L / 2), (PORT_OUT, cell.L / 2)]
        # get_stub_cell is shunt-first [Y][L*l2].
        return [(PORT_OUT, cell.L * cell.l2)]
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

    abcds = [
        np.asarray(cell.single_abcd(freqs_hz)).astype(np.complex128) for cell, _ in runs
    ]
    T = np.broadcast_to(np.eye(2, dtype=np.complex128), (n_f, 2, 2)).copy()
    for Tc, (_, n) in zip(abcds, runs):
        T = T @ np.linalg.matrix_power(Tc, n)

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
    J_p, J_m, L_nl, planes = [], [], [], []
    plane = 0
    for Tc, (cell, n) in zip(abcds, runs):
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

    pos = np.asarray(planes, dtype=np.float64)[None, :]
    phi_plus = np.stack(J_p, axis=-1) * np.exp(gamma_exact[:, None] * pos)
    phi_minus = np.stack(J_m, axis=-1) * np.exp(-gamma_exact[:, None] * pos)

    return {
        "Zb_plus": Zb_plus,
        "Zb_minus": Zb_minus,
        "W": W,
        "phi_plus": phi_plus,
        "phi_minus": phi_minus,
        "L_nl": np.asarray(L_nl, dtype=np.float64),
        "n_cells": n_cells,
        "positions": np.asarray(planes, dtype=np.float64),
        "gamma_exact": gamma_exact,
        "eig_residual": np.abs(psi_p[:, 1] * lam_f - 1),
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
# diagonalised). See ``Claude outputs/edge_basis_physics.tex``.


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
    T = np.broadcast_to(np.eye(2, dtype=np.complex128), (len(freqs_hz), 2, 2)).copy()
    for cell, n in expand_unit_cells(twpa):
        Tc = np.asarray(cell.single_abcd(freqs_hz)).astype(np.complex128)
        T = T @ np.linalg.matrix_power(Tc, n)
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
    from scipy.optimize import brentq

    def half_trace(f):
        T = supercell_matrix(twpa_ll, np.atleast_1d(f))
        return 0.5 * (T[:, 0, 0] + T[:, 1, 1]).real

    h_ref = np.sqrt(1.0 - e_ref**2)
    span = span_hz
    while span <= max_span_hz:
        fs = f_hz + np.linspace(-span, span, n_scan)
        fs = fs[fs > 0]
        h = half_trace(fs)
        inband = np.abs(h) < 1.0
        cross = np.flatnonzero(inband[1:] != inband[:-1])
        if cross.size:
            j = cross[np.argmin(np.abs(fs[cross] - f_hz))]
            f_edge = brentq(
                lambda f: abs(half_trace(f)[0]) - 1.0, fs[j], fs[j + 1], xtol=1.0
            )
            parity = int(np.sign(h[j] if not inband[j] else h[j + 1]))
            # Walk from the edge into the passband until |h| < h_ref.
            step = 1 if inband[j + 1] else -1
            k = j + 1 if step == 1 else j
            while 0 <= k < len(fs) and inband[k] and abs(h[k]) >= h_ref:
                k += step
            if 0 <= k < len(fs) and inband[k]:
                # |h| - h_ref changes sign between fs[k - step] (outside the
                # band or above h_ref) and fs[k].
                a, b = sorted((fs[k - step], fs[k]))
                f_ref = brentq(lambda f: abs(half_trace(f)[0]) - h_ref, a, b, xtol=1.0)
                return float(f_ref), float(f_edge), parity
        span *= 2
    raise ValueError(
        f"No stopband edge with a passband reaching split {e_ref} found around "
        f"{f_hz / 1e9:.4f} GHz."
    )


def _intracell_currents(
    twpa: TwoPortArray, freqs_hz: FloatArray, psi_p: np.ndarray, psi_m: np.ndarray
):
    """Currents of two boundary vectors propagated through one supercell.

    Same traversal as :func:`bloch_mode_tables`; returns the currents through
    every nonlinear inductor ``(F, n_nl)`` for each vector, the inductances
    and the positions (unit cells).
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
    J_p, J_m, L_nl, pos = _intracell_currents(twpa, f_hz, psi_p, psi_m)
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


def bloch_coefficients(
    relations: np.ndarray,
    unit_coeffs: np.ndarray,
    nl_coeff: float,
    omegas: np.ndarray,
    phi_plus: np.ndarray,
    phi_minus: np.ndarray,
    W: np.ndarray,
    L_nl: np.ndarray,
    n_cells: int,
) -> np.ndarray:
    """
    Per-frequency mixing coefficients in the Bloch basis (forward waves only).

    Parameters
    ----------
    relations : (R, 1+p) int
        Rows ``[target, input_1, ...]``; an input index ``>= n_modes`` means the
        complex conjugate of mode ``index - n_modes`` (kernel convention).
    unit_coeffs : (R,) complex
        RWA multiplicities divided by 4, i.e. ``prepare_relations_coefficients``
        output with ``epsilon = xi = 1``.
    nl_coeff : float
        ``epsilon`` (3WM) or ``xi`` (4WM).
    omegas, W : (F, n_modes)
        Angular frequencies and Bloch normalization of every mode.
    phi_plus, phi_minus : (F, n_modes, n_nl)
        Periodic current profiles through the nonlinear inductors.
    L_nl : (n_nl,)
        Nonlinear inductances.
    n_cells : int
        Unit cells per supercell.

    Returns
    -------
    (F, R) complex coefficients, already including the per-mode prefactor
    ``-2j omega_n / W_n`` (the kernel must not multiply by gamma again).
    """
    n_f, n_modes, _ = phi_plus.shape
    out = np.zeros((n_f, len(relations)), dtype=np.complex128)
    for r, rel in enumerate(relations):
        if unit_coeffs[r] == 0:
            continue
        n = rel[0]
        prod = L_nl[None, :] * phi_minus[:, n, :]
        for idx in rel[1:]:
            if idx >= n_modes:
                prod = prod * np.conj(phi_plus[:, idx - n_modes, :])
            else:
                prod = prod * phi_plus[:, idx, :]
        eta = prod.sum(axis=-1) / n_cells
        out[:, r] = unit_coeffs[r] * nl_coeff * (-2j * omegas[:, n] / W[:, n]) * eta
    return out


def cell_phase_mask(
    relations: np.ndarray, kappas: np.ndarray, n_cells: int, cell_phase_cutoff: float
) -> np.ndarray:
    """
    Mask of the forward mixing rows whose mismatch satisfies ``|dk| d < cutoff``.

    The Bloch couplings are cell averages (G = 0 Fourier component of the
    intracell profiles); a term whose phase mismatch per supercell is not small
    compared with pi is closer to a quasi-phase-matched harmonic G != 0 and is
    not described by them. The condition depends only on ``|dk|``, which is the
    same for all partner terms of a process, so they are kept or dropped
    together.

    Parameters
    ----------
    relations : (R, 1+p) int
        Rows ``[target, input_1, ...]``; inputs ``>= n_modes`` are conjugated.
    kappas : (F, n_modes) float
        Kernel wavenumbers (``Im`` of the kernel gammas), per unit cell.
    n_cells : int
        Unit cells per supercell.
    cell_phase_cutoff : float

    Returns
    -------
    (F, R) bool
    """
    n_modes = kappas.shape[1]
    dk = np.zeros((kappas.shape[0], len(relations)))
    for r, rel in enumerate(relations):
        dk[:, r] = -kappas[:, rel[0]]
        for idx in rel[1:]:
            if idx >= n_modes:
                dk[:, r] -= kappas[:, idx - n_modes]
            else:
                dk[:, r] += kappas[:, idx]
    return np.abs(dk) * n_cells < cell_phase_cutoff


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


def edge_guard_mode_weights(W: np.ndarray, lo_deg: float, hi_deg: float) -> np.ndarray:
    """
    Stopband weights of each mode from its reactive angle (forward-only model).

    1 below ``lo_deg``, 0 above ``hi_deg``, smoothstep in between (a hard switch
    at ``hi_deg`` if ``hi_deg <= lo_deg``). Zero inside a whole stopband.
    """
    ang = reactive_angle_deg(W)
    if hi_deg <= lo_deg:
        return np.where(ang < hi_deg, 1.0, 0.0)
    t = np.clip((ang - lo_deg) / (hi_deg - lo_deg), 0.0, 1.0)
    return 1.0 - t * t * (3.0 - 2.0 * t)


def edge_proximity_mode_weights(split: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """
    Stopband-edge weights of each mode from its Bloch splitting.

    ``split = |sinh(gamma d)|`` (see :func:`bloch_mode_tables`) is 0 at every
    band edge and grows on both sides, in the passband and inside the gap, so
    the weight vanishes only in a window around each edge: 0 for
    ``split <= lo``, 1 for ``split >= hi``, smoothstep in between (a hard switch
    at ``hi`` if ``hi <= lo``).
    """
    split = np.asarray(split, dtype=np.float64)
    if hi <= lo:
        return np.where(split >= hi, 1.0, 0.0)
    t = np.clip((split - lo) / (hi - lo), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


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
    mode_weights : (F, n_modes) float

    Returns
    -------
    (F, R) float
    """
    n_f, n_modes = mode_weights.shape
    out = np.ones((n_f, len(relations)), dtype=np.float64)
    for r, rel in enumerate(relations):
        for idx in rel:
            out[:, r] *= mode_weights[:, int(idx) % n_modes]
    return out


# --------------------------------------------------------------------------- #
# Nonlinear part: CME kernel and parallel solver
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
    ),
    cache=True,
)
def bloch_no_reflections_cmes(
    x,
    currents,
    alphas,
    kappas,
    relations_3wm,
    relations_4wm,
    coeffs_3wm,
    coeffs_4wm,
):
    """
    RHS of the no-reflections Bloch CMEs for a uniform device.

    Same structure as ``no_reflections_cmes``, but the per-mode prefactor is
    already folded into ``coeffs_*`` (see :func:`bloch_coefficients`).
    """
    num_modes = len(currents)
    derivs = np.zeros(num_modes, nb.complex128)
    exp_pos = np.exp(1j * kappas * x)

    fields = np.zeros(2 * num_modes, dtype=np.complex128)
    for i in range(num_modes):
        fields[i] = currents[i] * exp_pos[i]
        fields[i + num_modes] = np.conj(fields[i])

    for i, (idx, idx1, idx2) in enumerate(relations_3wm):
        derivs[idx] += coeffs_3wm[i] * fields[idx1] * fields[idx2]
    for i, (idx, idx1, idx2, idx3) in enumerate(relations_4wm):
        derivs[idx] += coeffs_4wm[i] * fields[idx1] * fields[idx2] * fields[idx3]
    derivs = derivs / exp_pos + alphas * currents
    return derivs


@nb.njit(parallel=True, cache=True)
def _bloch_no_reflections_solve_uniform(
    x_array,
    y0_broadcast,
    gammas_array,
    relations_3wm,
    relations_4wm,
    coeffs_3wm_table,
    coeffs_4wm_table,
):
    """Parallel solve; coefficient tables have shape (n_freq, n_relations)."""
    n_freq = y0_broadcast.shape[0]
    n_modes = y0_broadcast.shape[1]
    x_span = (x_array[0], x_array[-1])
    I_tuples = np.empty((n_freq, n_modes, len(x_array)), dtype=nb.complex128)

    for i in nb.prange(n_freq):
        I_tuples[i] = rk45_solve(
            bloch_no_reflections_cmes,
            x_span,
            y0_broadcast[i],
            args=(
                gammas_array[i].real,
                gammas_array[i].imag,
                relations_3wm,
                relations_4wm,
                coeffs_3wm_table[i],
                coeffs_4wm_table[i],
            ),
            atol=SOLVER_ATOL,
            rtol=SOLVER_RTOL,
            max_num_steps=SOLVER_MAX_STEPS,
            t_eval=x_array,
            first_step=SOLVER_FIRST_STEP,
        )

    return I_tuples


def bloch_no_reflections_cmes_solve(
    x_array: FloatArray,
    y0_broadcast: ComplexArray,
    gammas_array: ComplexArray,
    relations_3wm: np.ndarray,
    relations_4wm: np.ndarray,
    coeffs_3wm_table: ComplexArray,
    coeffs_4wm_table: ComplexArray,
) -> ComplexArray:
    """
    Solve the no-reflections Bloch CMEs for every signal frequency in parallel.

    Returns slow-envelope amplitudes of shape (n_freq, n_modes, n_x); apply the
    phase carrier in the caller as for the standard model.
    """
    if gammas_array.ndim != 2:
        raise NotImplementedError("Bloch CMEs support only non-segmented devices.")
    return _bloch_no_reflections_solve_uniform(
        np.ascontiguousarray(x_array, dtype=np.float64),
        np.ascontiguousarray(y0_broadcast, dtype=np.complex128),
        np.ascontiguousarray(gammas_array, dtype=np.complex128),
        np.ascontiguousarray(relations_3wm, dtype=np.int64),
        np.ascontiguousarray(relations_4wm, dtype=np.int64),
        np.ascontiguousarray(coeffs_3wm_table, dtype=np.complex128),
        np.ascontiguousarray(coeffs_4wm_table, dtype=np.complex128),
    )
