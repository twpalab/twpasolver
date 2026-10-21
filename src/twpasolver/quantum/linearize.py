"""
Linearised Bloch CMEs around a classical pump, in canonical flux variables.

Physical coordinate x (unit cells) for both directions. For a weak mode ``n`` in
direction ``sigma = +-1`` the projected equation (derivation, Eq. projected) is

    dA_n^sigma/dx = -sigma (j omega_n / W_n) < L Pi_n phi_n^{-sigma} > e^{sigma j k_n x}

with the current through an element ``I_r = A_r^+ phi_r^+ e^{-jk_r x} +
A_r^- phi_r^- e^{+jk_r x}``. Expanding ``Pi_n`` into direction variants and the
periodic factor into reciprocal-lattice harmonics, a kept term reads

    dA_n^sigma/dx  +=  c_q  e^{j r_q x}  prod_{pump inputs} A_p^{tau_p}(x)^{(*)}  A_w^{tau_w (*)}

with ``c_q = sigma * (-2j omega_n / W_n) * base_q * w_q * Fhat_{q,m}``,
``Fhat_{q,m} = (1/d) sum_j L_j phi_n^{-sigma}(x_j) prod_r phi_r^{tau_r}(x_j)^{(*)}
e^{-j G_m x_j}`` and residual phase ``r_q = Phi_q + G_m``,
``Phi_q = sigma k_n - sum_r c_r tau_r k_r`` (``c_r = -1`` for conjugated inputs).
Selection: ``|r_q| < kappa_cutoff * min(k_proc, pi / d)``, ``|m| <= m_max``
(identical to ``_filter_wm_terms_bloch``, partners kept together).

Only rows with a weak target and exactly one weak input enter the linear
generator. The flux variables ``a = A sqrt(W / 4 hbar omega)`` (lossless ``W``)
turn the generator into ``K`` with ``K^dag Sigma + Sigma K = 0``,
``Sigma = diag(+1, -1, -1, +1)`` on ``v = (a^+, a^-, a^{+*}, a^{-*})``.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np
from scipy.interpolate import CubicSpline

from twpasolver.quantum.background import Background


@dataclass
class LinearTerms:
    """Kept linear terms of one frequency point (current basis)."""

    active: list[int]  # mode-array indices of the active weak modes
    tgt: np.ndarray  # (T,) row in the 2M non-conjugated block (a^+, a^-)
    src: np.ndarray  # (T,) column in the 4M vector (a^+, a^-, a^+*, a^-*)
    coeff: np.ndarray  # (T,) complex
    phase: np.ndarray  # (T,) residual phase r_q (term ~ e^{j r_q x})
    slots: np.ndarray  # (T, 3) indices into the pump-field vector (padding -> 1)
    scale: np.ndarray  # (M,) sqrt(W / omega) of the active modes
    n_driven: int  # weak targets driven by pump-only rows (diagnostic)


def active_weak_modes(bg: Background, i: int, w_tol: float = 1e-6) -> list[int]:
    """Weak modes that are canonical travelling waves at frequency point ``i``."""
    out = []
    for n in bg.weak:
        W = bg.W[i, n]
        ok = (
            bg.mode_weights[i, n] > 0
            and bg.freqs_signed[i, n] > 0
            and np.isfinite(W)
            and W.real > 0
            and abs(W.imag) <= w_tol * abs(W)
        )
        if ok:
            out.append(n)
    return out


def linear_terms(
    bg: Background, i: int, active: list[int] | None = None
) -> LinearTerms:
    """Collect the kept linear terms of frequency point ``i``."""
    if active is None:
        active = active_weak_modes(bg, i)
    n_modes = len(bg.mode_labels)
    M = len(active)
    Np = len(bg.pump)
    wpos = {n: j for j, n in enumerate(active)}
    ppos = {p: j for j, p in enumerate(bg.pump)}
    weak_all = set(bg.weak)
    pad = 4 * Np

    k = bg.k[i]
    d = bg.n_cells
    g_unit = 2 * np.pi / d
    cutoff = bg.fb_config.kappa_cutoff
    m_max = bg.fb_config.m_max_fixed
    k_edge = np.pi / d  # zone edge: cap of the reference wavenumber
    prefac = -2j * bg.omegas[i] / bg.W[i]
    phi = {+1: bg.phi_plus[i], -1: bg.phi_minus[i]}
    Lnl = bg.L_nl
    pos = bg.positions

    tgt, src, coeff, phase, slots = [], [], [], [], []
    n_driven = 0
    for rels, bases, pw in (
        (bg.relations_3wm, bg.base_3wm, bg.pw3[i]),
        (bg.relations_4wm, bg.base_4wm, bg.pw4[i]),
    ):
        for r, rel in enumerate(rels):
            base = bases[r]
            if base == 0 or pw[r] == 0:
                continue
            n = int(rel[0])
            if n not in weak_all:
                continue
            inputs = [(int(v) % n_modes, int(v) >= n_modes) for v in rel[1:]]
            weak_in = [q for q, (m, _) in enumerate(inputs) if m in weak_all]
            if len(weak_in) == 0:
                n_driven += 1
                continue
            if len(weak_in) != 1:
                continue
            if n not in wpos or inputs[weak_in[0]][0] not in wpos:
                continue  # inactive weak mode: its processes are dropped
            k_proc = max(abs(k[n]), *(abs(k[m]) for m, _ in inputs))
            for sigma in (+1, -1):
                for taus in product((+1, -1), repeat=len(inputs)):
                    Phi = sigma * k[n]
                    for (m, cj), tau in zip(inputs, taus):
                        Phi -= (-1 if cj else 1) * tau * k[m]
                    mg = -int(np.round(Phi * d / (2 * np.pi)))
                    if abs(mg) > m_max:
                        continue
                    res = Phi + mg * g_unit
                    if not abs(res) < min(cutoff * min(k_proc, k_edge), k_edge):
                        continue
                    F = Lnl * phi[-sigma][n]
                    for (m, cj), tau in zip(inputs, taus):
                        f = phi[tau][m]
                        F = F * (np.conj(f) if cj else f)
                    Fhat = np.sum(F * np.exp(-1j * mg * g_unit * pos)) / d
                    c = sigma * prefac[n] * base * pw[r] * Fhat
                    sl = [pad, pad, pad]
                    q_slot = 0
                    for q, ((m, cj), tau) in enumerate(zip(inputs, taus)):
                        if q == weak_in[0]:
                            col = wpos[m] + (0 if tau > 0 else M) + (2 * M if cj else 0)
                        else:
                            sl[q_slot] = (
                                ppos[m] + (0 if tau > 0 else Np) + (2 * Np if cj else 0)
                            )
                            q_slot += 1
                    tgt.append(wpos[n] + (0 if sigma > 0 else M))
                    src.append(col)
                    coeff.append(c)
                    phase.append(res)
                    slots.append(sl)
    scale = np.sqrt(bg.W[i, active].real / bg.omegas[i, active])
    return LinearTerms(
        active=list(active),
        tgt=np.asarray(tgt, dtype=np.int64),
        src=np.asarray(src, dtype=np.int64),
        coeff=np.asarray(coeff, dtype=np.complex128),
        phase=np.asarray(phase, dtype=np.float64),
        slots=np.asarray(slots, dtype=np.int64).reshape(-1, 3),
        scale=scale,
        n_driven=n_driven,
    )


class PumpField:
    """Cubic interpolation of the pump Bloch envelopes of one frequency point."""

    def __init__(self, bg: Background, i: int):
        x = bg.x_pump
        P = bg.pump
        self.Np = len(P)
        data = np.concatenate([bg.pump_fwd[i, P], bg.pump_bwd[i, P]], axis=0)
        self._re = CubicSpline(x, data.real, axis=1)
        self._im = CubicSpline(x, data.imag, axis=1)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Field vector ``[A^+, A^-, A^+*, A^-*, 1]`` at positions ``x``: (n_x, 4Np+1)."""
        x = np.atleast_1d(x)
        A = (self._re(x) + 1j * self._im(x)).T  # (n_x, 2Np)
        return np.concatenate([A, np.conj(A), np.ones((len(x), 1))], axis=1)


def generator(terms: LinearTerms, pf: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    Flux-basis generator ``K(x)`` on ``v = (a^+, a^-, a^{+*}, a^{-*})``.

    Parameters
    ----------
    terms : LinearTerms
    pf : (n_x, 4Np+1) pump-field vectors at ``x``
    x : (n_x,) positions

    Returns
    -------
    (n_x, 4M, 4M) complex
    """
    M = len(terms.active)
    nx = len(x)
    top = np.zeros((nx, 2 * M * 4 * M), dtype=np.complex128)
    if len(terms.coeff):
        vals = (
            terms.coeff[None, :]
            * np.exp(1j * np.outer(x, terms.phase))
            * pf[:, terms.slots[:, 0]]
            * pf[:, terms.slots[:, 1]]
            * pf[:, terms.slots[:, 2]]
        )
        flat = terms.tgt * (4 * M) + terms.src
        scatter = np.zeros((len(flat), 8 * M * M))
        scatter[np.arange(len(flat)), flat] = 1.0
        top = vals @ scatter
    top = top.reshape(nx, 2 * M, 4 * M)
    s = terms.scale
    s_row = np.r_[s, s]
    s_col = np.r_[s, s, s, s]
    top = top * s_row[None, :, None] / s_col[None, None, :]
    K1 = top[:, :, : 2 * M]
    K2 = top[:, :, 2 * M :]
    return np.concatenate(
        [
            np.concatenate([K1, K2], axis=2),
            np.concatenate([K2.conj(), K1.conj()], axis=2),
        ],
        axis=1,
    )


def sigma_metric(M: int) -> np.ndarray:
    """``Sigma = diag(+1, -1, -1, +1)`` (blocks of M) on ``(a^+, a^-, a^{+*}, a^{-*})``."""
    return np.diag(np.r_[np.ones(M), -np.ones(M), -np.ones(M), np.ones(M)])


def hamiltonian_defect(K: np.ndarray) -> float:
    """``max ||K^dag Sigma + Sigma K|| / max ||K||`` over the stack (0 if K = 0)."""
    M = K.shape[-1] // 4
    S = sigma_metric(M)
    R = np.swapaxes(K.conj(), -1, -2) @ S + S @ K
    scale = np.abs(K).max()
    return float(np.abs(R).max() / scale) if scale > 0 else 0.0
