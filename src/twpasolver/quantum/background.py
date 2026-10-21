"""
Classical background of the linearised quantum model.

Everything the quantum layers need from the classical Bloch framework, built
for a fixed pump and a sweep of signal frequencies:

* the split of the mode array into *weak* modes (the signal and every mode
  that depends on it: idler, sidebands, signal harmonics) and *pump* modes
  (the pump and its harmonics, treated classically);
* Bloch tables of the **lossless** supercell (``delta`` set to zero in every
  cell): real ``W`` in every passband, ``phi^- = (phi^+)^*``, so that the flux
  variables ``a = A sqrt(W / 4 hbar omega)`` are canonical;
* the attenuation ``alpha`` of every mode from the **lossy** cell, used only by
  the loss channels of :mod:`twpasolver.quantum.gaussian`;
* the stopband-edge weights of the classical solver (from the lossy ``W``);
* the pump background: a forward-backward Bloch solve with the signal set to
  zero, run with the *same* lossless tables and the lossy ``alpha`` that the
  quantum model uses (pump reflections, harmonics and depletion included),
  once per sweep (the pump does not depend on the signal frequency), with the
  integrator step capped at the grid spacing.

Pump envelopes are returned in the physical coordinate as the Bloch amplitudes
``A^+(x)``, ``A^-(x)`` (boundary currents with the lossless propagation factored
out, loss kept in the envelope).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from twpasolver.analysis import prepare_relations_coefficients
from twpasolver.cmes import solver_config
from twpasolver.cmes.bloch import (  # noqa: F401  (lossless_copy re-exported)
    bloch_mode_tables,
    edge_guard_mode_weights,
    lossless_copy,
    process_weights,
)
from twpasolver.cmes.bloch_fb import bloch_fb_cmes_solve
from twpasolver.cmes.solver_config import FBSolverConfig
from twpasolver.logger import log


def has_loss(model: Any) -> bool:
    """True if any cell of ``model`` has a nonzero ``delta``."""
    if getattr(model, "delta", 0.0):
        return True
    return any(has_loss(c) for c in getattr(model, "cells", None) or [])


@dataclass
class Background:
    """Classical data shared by all quantum layers (arrays over signal frequencies)."""

    signal_freqs: np.ndarray
    pump_freq: float
    mode_labels: list[str]
    weak: list[int]  # mode indices of the weak modes; weak[0] is the signal
    pump: list[int]  # mode indices of the classical modes
    signal_mode: str
    idler_mode: str | None
    omegas: np.ndarray  # (F, n_modes) rad/s, |omega|
    freqs_signed: np.ndarray  # (F, n_modes) Hz, as given by the mode array
    k: np.ndarray  # (F, n_modes) lossless Bloch wavenumber per unit cell
    alpha: np.ndarray  # (F, n_modes) lossy attenuation per unit cell
    W: np.ndarray  # (F, n_modes) lossless reciprocity invariant
    W_lossy: np.ndarray  # (F, n_modes)
    Zb_plus: np.ndarray  # (F, n_modes) lossless
    Zb_minus: np.ndarray  # (F, n_modes) lossless
    phi_plus: np.ndarray  # (F, n_modes, n_nl) lossless
    phi_minus: np.ndarray  # (F, n_modes, n_nl) lossless
    L_nl: np.ndarray
    positions: np.ndarray
    n_cells: int
    L: float  # device length in unit cells
    relations_3wm: np.ndarray
    relations_4wm: np.ndarray
    base_3wm: np.ndarray  # RWA multiplicity/4 * epsilon
    base_4wm: np.ndarray  # RWA multiplicity/4 * xi
    pw3: np.ndarray  # (F, n_rel3) process weights (edge guard)
    pw4: np.ndarray  # (F, n_rel4)
    mode_weights: np.ndarray  # (F, n_modes)
    x_pump: np.ndarray  # (n_x,) grid of the pump solve
    pump_fwd: np.ndarray  # (F, n_modes, n_x) A^+(x) (zero for weak modes)
    pump_bwd: np.ndarray  # (F, n_modes, n_x) A^-(x)
    abcd_in: np.ndarray  # (F, n_modes, 4)
    abcd_out: np.ndarray  # (F, n_modes, 4)
    Ze: float
    fb_config: FBSolverConfig
    extra: dict[str, Any] = field(default_factory=dict)


def build_background(
    analysis,
    signal_freqs,
    pump: float | None = None,
    Ip0: float | None = None,
    mode_array_config: str = "basic_3wm",
    signal_mode: str = "s",
    pump_mode: str = "p",
    idler_mode: str | None = "i",
    thin: float | None = None,
    pump_tol: float = 1e-8,
    pump_passes: int = 60,
    fb_config: FBSolverConfig | None = None,
    pump_inputs_fwd: dict[str, complex] | None = None,
    pump_inputs_bwd: dict[str, complex] | None = None,
    edge_guard: bool = True,
) -> Background:
    """
    Classical background for the linearised quantum model.

    Parameters
    ----------
    analysis : TWPAnalysis
        Non-segmented device; cells must carry their loss as ``delta``.
    signal_freqs : array
        Signal frequencies, in the analysis unit.
    pump, Ip0 : float, optional
        Pump frequency and amplitude (defaults as in ``TWPAnalysis.gain``).
    pump_inputs_fwd, pump_inputs_bwd : dict, optional
        Extra classical inputs ``{label: current}`` at x=0 / x=L (e.g. a second
        pump); the pump mode gets ``Ip0`` at x=0 unless overridden.
    edge_guard : bool
        Apply the stopband-edge weights of the classical solver.
    thin : float, optional
        Grid spacing (unit cells) of the pump solve; default one supercell.
    pump_tol, pump_passes : float, int
        Fixed-point tolerance and pass limit of the pump forward-backward solve.
        The pump-only problem has no weak-mode noise floor, so it is converged
        far below the classical default (reflected pumps of mismatched devices
        otherwise carry ~1% errors into the gain).
    """
    fb_config = fb_config if fb_config is not None else FBSolverConfig()
    if analysis._is_segmented():
        raise NotImplementedError("Quantum Bloch model: non-segmented devices only.")
    if Ip0 is not None:
        analysis.twpa.Ip0 = Ip0
    analysis.update_base_data()
    if pump is None:
        pump = analysis.data["optimal_pump_freq"]
    signal_freqs = np.atleast_1d(np.asarray(signal_freqs, dtype=np.float64))

    mode_array = analysis.get_mode_array(mode_array_config)
    cme = analysis._prepare_cme_inputs(
        mode_array, signal_freqs, pump, signal_mode, pump_mode
    )
    labels = cme["mode_labels"]
    n_freq, n_modes = cme["n_freq"], cme["n_modes"]
    mode_params = cme["mode_params"]

    weak_labels = set(mode_array.get_dependent_modes(signal_mode)) | {signal_mode}
    weak = [labels.index(signal_mode)] + [
        j for j, l in enumerate(labels) if l in weak_labels and l != signal_mode
    ]
    pump_idx = [j for j, l in enumerate(labels) if l not in weak_labels]

    freqs_signed = np.stack([mode_params[m]["freqs"] for m in labels], axis=-1)
    freqs_hz = np.abs(freqs_signed) * analysis.unit_multiplier
    omegas = 2 * np.pi * freqs_hz
    alpha = np.stack([mode_params[m]["alpha"] for m in labels], axis=-1)
    k_interp = np.stack([mode_params[m]["k"] for m in labels], axis=-1)
    gamma_ref = (alpha + 1j * k_interp).ravel()

    # Lossless tables define the canonical basis; lossy W only feeds the guard.
    twpa_ll = lossless_copy(analysis.twpa)
    tab = bloch_mode_tables(twpa_ll, freqs_hz.ravel(), gamma_ref)
    tab_lossy = bloch_mode_tables(analysis.twpa, freqs_hz.ravel(), gamma_ref)
    n_nl = tab["L_nl"].size
    shp = (n_freq, n_modes)
    W = tab["W"].reshape(shp)
    W_lossy = tab_lossy["W"].reshape(shp)
    k = tab["gamma_exact"].imag.reshape(shp)
    phi_plus = tab["phi_plus"].reshape(n_freq, n_modes, n_nl)
    phi_minus = tab["phi_minus"].reshape(n_freq, n_modes, n_nl)
    Zb_plus = tab["Zb_plus"].reshape(shp)
    Zb_minus = tab["Zb_minus"].reshape(shp)

    _, _, unit_3wm, unit_4wm = prepare_relations_coefficients(
        mode_array.get_rwa_terms(power=2), mode_array.get_rwa_terms(power=3), 1.0, 1.0
    )
    base_3wm = unit_3wm * analysis.twpa.epsilon
    base_4wm = unit_4wm * analysis.twpa.xi

    # Classical inputs (pump only).
    y0_fwd = np.zeros(n_modes, dtype=np.complex128)
    y0_bwd = np.zeros(n_modes, dtype=np.complex128)
    y0_fwd[labels.index(pump_mode)] = analysis.twpa.Ip0
    for lab, val in (pump_inputs_fwd or {}).items():
        y0_fwd[labels.index(lab)] = val
    for lab, val in (pump_inputs_bwd or {}).items():
        y0_bwd[labels.index(lab)] = val
    if np.any(y0_fwd[weak] != 0) or np.any(y0_bwd[weak] != 0):
        raise ValueError("Classical inputs must not drive weak modes.")
    y0f = np.repeat(y0_fwd[None], n_freq, axis=0)
    y0b = np.repeat(y0_bwd[None], n_freq, axis=0)

    # Edge guard (same weights as the classical bloch model).
    injected = (np.abs(y0_fwd) > 0) | (np.abs(y0_bwd) > 0)
    guard = solver_config.BLOCH_EDGE_GUARD_DEG if edge_guard else None
    if guard is None:
        mode_weights = np.ones(shp)
    else:
        mode_weights = edge_guard_mode_weights(W_lossy, *guard)
        mode_weights[:, injected] = 1.0
    pw3 = process_weights(cme["relations_3wm"], mode_weights)
    pw4 = process_weights(cme["relations_4wm"], mode_weights)

    Zout_in, Zout_out, abcd_in, abcd_out = analysis._build_port_tables(
        mode_params, labels, n_freq
    )
    Ze = float(analysis.port_Z_ext)

    # Pump background with the quantum model's tables: k lossless, alpha lossy.
    # The pump-only problem does not depend on the signal frequency (weak
    # modes are zero), so it is solved once on the first row and broadcast.
    N_tot = analysis.twpa.N_tot
    L = float(N_tot)
    # Pump grid: one supercell by default. The integrator step is capped at the
    # grid spacing so that the stored envelopes (linear interpolation between
    # accepted steps) stay accurate for a pump rotating under SPM/XPM.
    spacing = float(tab["n_cells"] if thin is None else thin)
    x = np.linspace(0, N_tot, max(int(round(N_tot / spacing)), 8) + 1)
    gammas_q = -alpha - 1j * k
    prefac = -2j * omegas / W
    r0 = slice(0, 1)
    raw = bloch_fb_cmes_solve(
        x,
        y0f[r0],
        y0b[r0],
        gammas_q[r0],
        cme["relations_3wm"],
        cme["relations_4wm"],
        base_3wm,
        base_4wm,
        prefac[r0],
        phi_plus[r0],
        phi_minus[r0],
        tab["L_nl"],
        tab["positions"],
        tab["n_cells"],
        Zb_plus[r0],
        Zb_minus[r0],
        passes=max(fb_config.passes, pump_passes),
        kappa_cutoff=fb_config.kappa_cutoff,
        m_max=fb_config.m_max_fixed,
        convergence_threshold=min(fb_config.convergence_threshold, pump_tol),
        anderson_depth=fb_config.anderson_depth,
        process_weights_3wm=pw3[r0],
        process_weights_4wm=pw4[r0],
        Ze=Ze,
        Zout_in=Zout_in[r0],
        Zout_out=Zout_out[r0],
        abcd_in=abcd_in[r0],
        abcd_out=abcd_out[r0],
        max_step=float(x[1] - x[0]),
    )
    rho_in = (Zb_minus - Zout_in) / (Zb_plus + Zout_in)
    rho_out = (Zb_plus - Zout_out) / (Zb_minus + Zout_out)
    A_, B_, C_, D_ = (abcd_in[..., j] for j in range(4))
    tof_in = (A_ * D_ - B_ * C_) / (A_ + C_ * Ze)
    A_, B_, C_, D_ = (abcd_out[..., j] for j in range(4))
    apcz_out = A_ + C_ * Ze
    tof_out = (A_ * D_ - B_ * C_) / apcz_out
    raw_f = np.array(raw[0])
    raw_b = np.array(raw[1])
    raw_f[:, -1] /= (1.0 + rho_out[0]) * (-tof_out[0])  # undo the outcoupling store
    raw_b[:, -1] /= (1.0 + rho_in[0]) * tof_in[0]
    B_f, B_b = raw_f, raw_b
    # Physical Bloch amplitudes: A+(x) = B_fwd(x), A-(x) = B_bwd(L - x) e^{-jkL}.
    pump_fwd = np.repeat(B_f[None], n_freq, axis=0)
    pump_bwd = np.repeat(
        (B_b[:, ::-1] * np.exp(-1j * k[0] * L)[:, None])[None], n_freq, axis=0
    )
    pump_fwd[:, weak, :] = 0.0
    pump_bwd[:, weak, :] = 0.0

    if idler_mode is not None and idler_mode not in labels:
        log.warning("Idler mode %s not in the mode array; ignored.", idler_mode)
        idler_mode = None

    return Background(
        signal_freqs=signal_freqs,
        pump_freq=float(pump),
        mode_labels=labels,
        weak=weak,
        pump=pump_idx,
        signal_mode=signal_mode,
        idler_mode=idler_mode,
        omegas=omegas,
        freqs_signed=freqs_signed,
        k=k,
        alpha=alpha,
        W=W,
        W_lossy=W_lossy,
        Zb_plus=Zb_plus,
        Zb_minus=Zb_minus,
        phi_plus=phi_plus,
        phi_minus=phi_minus,
        L_nl=tab["L_nl"],
        positions=tab["positions"],
        n_cells=int(tab["n_cells"]),
        L=L,
        relations_3wm=cme["relations_3wm"],
        relations_4wm=cme["relations_4wm"],
        base_3wm=base_3wm,
        base_4wm=base_4wm,
        pw3=pw3,
        pw4=pw4,
        mode_weights=mode_weights,
        x_pump=x,
        pump_fwd=pump_fwd,
        pump_bwd=pump_bwd,
        abcd_in=abcd_in,
        abcd_out=abcd_out,
        Ze=Ze,
        fb_config=fb_config,
        extra={"Zout_in": Zout_in, "Zout_out": Zout_out, "y0_fwd": y0_fwd},
    )
