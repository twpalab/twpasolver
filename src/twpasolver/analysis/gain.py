"""Gain models: the forward-only ``simple`` CMEs and the forward-backward ``bloch`` CMEs."""

import enum
from typing import Any

import numpy as np

from twpasolver.analysis.ports import Ports
from twpasolver.analysis.results import GainResult
from twpasolver.cmes.base import no_reflections_cmes_solve
from twpasolver.cmes.bloch_fb import bloch_fb_segmented_cmes_solve
from twpasolver.cmes.prep import CMEProblem, fb_stopbands, prepare_bloch_segmented
from twpasolver.cmes.solver_config import BlochSolverConfig, RKSolverConfig
from twpasolver.mathutils import to_dB


class GainModel(str, enum.Enum):
    """
    Available CME models for gain analysis.

    SIMPLE : "simple"
        Traditional coupled mode equations of a homogeneous line, forward waves
        only. The supercell enters only through the Bloch wavenumber; the
        couplings are those of a uniform medium (scaled by k), without the
        intracell current profiles, so they are not exact for loaded lines,
        especially near stopbands. Ignores reflections and the port/connector
        models. Fast; for quick sweeps and comparisons with the literature.
        Segmented devices are interpolated linearly between segment midpoints
        (a smooth taper).

    BLOCH : "bloch"
        Forward-backward model in the Bloch-mode basis: exact Bloch couplings
        with the intracell current profiles, counter-propagating waves and
        reflections, exact boundary conditions with the port/connector models,
        fixed edge basis for modes close to a stopband edge. Segmented devices
        (whole supercells per segment) are solved on per-segment Bloch modes
        joined by exact junction conditions. Configured by ``BlochSolverConfig``.
    """

    SIMPLE = "simple"
    BLOCH = "bloch"


def _ratio_db(out: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """``20 log10 |out / ref|``, NaN where the reference input is zero."""
    out = np.asarray(out)
    ref = np.broadcast_to(np.asarray(ref), out.shape)
    res = np.full(out.shape, np.nan)
    ok = np.abs(ref) > 0
    res[ok] = to_dB(np.abs(out[ok] / ref[ok]))
    return res


def port_gains(a_in, a_out, b_in, b_out) -> dict[str, Any]:
    """S21, S12, S11 and S22 in dB from the external currents (``a`` in, ``b`` out)."""
    return {
        "gain_db": _ratio_db(b_out, a_in),
        "gain_db_12": _ratio_db(b_in, a_out),
        "gain_db_11": _ratio_db(b_in, a_in),
        "gain_db_22": _ratio_db(b_out, a_out),
    }


def _drop_segment_axis(bloch: dict[str, Any], n_freq: int) -> dict[str, Any]:
    """
    Bloch tables of a one-segment solve with the segment axis removed.

    Uniform devices are solved as one segment; their diagnostics keep the
    shapes of a uniform device: (n_freq, n_modes, ...) per-mode tables,
    (n_nl,) / (n_rel,) intracell tables and an integer ``n_cells``.
    """
    out = {}
    for key, val in bloch.items():
        if key in ("L_nl", "positions", "base_3wm", "base_4wm"):
            out[key] = np.asarray(val)[0]
        elif key == "n_cells":
            out[key] = int(np.asarray(val)[0])
        elif (
            isinstance(val, np.ndarray)
            and val.ndim >= 3
            and val.shape[0] == n_freq
            and val.shape[1] == 1
        ):
            out[key] = val[:, 0]
        else:
            out[key] = val
    return out


def _phase_carrier(problem: CMEProblem, x: np.ndarray) -> np.ndarray:
    """
    ``exp(i Phi_m(x))``, ``Phi_m = int Im(gamma_m) dx = -int k_m dx``, of the
    ``simple`` kernel, shape (n_freq, n_modes, n_x).

    ``k_m(x)`` is piecewise-linear on the segment midpoints (constant for one
    segment); the cumulative trapezoid rule is exact for that interpolant.
    """
    kappa = problem.gammas.imag
    kappa_at_x = np.array(
        [
            [
                np.interp(x, problem.seg_midpoints, kappa[i, j])
                for j in range(kappa.shape[1])
            ]
            for i in range(kappa.shape[0])
        ]
    )
    dphi = 0.5 * (kappa_at_x[..., :-1] + kappa_at_x[..., 1:]) * np.diff(x)
    phi = np.concatenate(
        [np.zeros((*kappa.shape[:2], 1)), np.cumsum(dphi, axis=-1)], -1
    )
    return np.exp(1j * phi)


def solve_simple(
    problem: CMEProblem,
    y0: np.ndarray,
    x: np.ndarray,
    rk_config: RKSolverConfig,
    **meta,
) -> GainResult:
    """
    Forward-only CMEs; ``y0`` (n_freq, n_modes) are the currents injected at x=0.

    No ports: ``b_out`` is the current at x=L, nothing comes back (``b_in`` NaN).
    """
    I_fwd = no_reflections_cmes_solve(
        x,
        y0,
        problem.gammas,
        problem.relations_3wm,
        problem.relations_4wm,
        problem.coeffs_3wm,
        problem.coeffs_4wm,
        segment_midpoints=problem.seg_midpoints,
        rk_config=rk_config,
    ) * _phase_carrier(problem, x)
    ports = {
        "a_in": y0,
        "a_out": np.zeros_like(y0),
        "b_in": np.full_like(y0, np.nan),
        "b_out": I_fwd[..., -1],
    }
    s = problem.mode_labels.index(meta["signal_mode"])
    return GainResult(
        model=GainModel.SIMPLE.value,
        x=x,
        mode_labels=problem.mode_labels,
        I_fwd=I_fwd,
        I_bwd=np.zeros_like(I_fwd),
        **ports,
        **port_gains(*(ports[k][:, s] for k in ("a_in", "a_out", "b_in", "b_out"))),
        converged=np.isfinite(I_fwd).all(axis=(1, 2)),
        **meta,
    )


def solve_bloch(
    problem: CMEProblem,
    y0_fwd: np.ndarray,
    y0_bwd: np.ndarray,
    x: np.ndarray,
    ports: Ports,
    bloch_config: BlochSolverConfig,
    rk_config: RKSolverConfig,
    **meta,
) -> GainResult:
    """
    Forward-backward CMEs in the Bloch basis; ``y0_fwd`` / ``y0_bwd``
    (n_freq, n_modes) are the currents incident from the external lines at x=0
    and x=L. Amplitudes are +x-referenced boundary currents of the forward and
    backward Bloch modes; see :mod:`twpasolver.cmes.bloch_fb`.
    """
    bloch = prepare_bloch_segmented(problem)
    Zout_in, Zout_out, abcd_in, abcd_out = ports.tables(problem.freqs_hz)
    injected = np.any(np.abs(y0_fwd) > 0, axis=0) | np.any(np.abs(y0_bwd) > 0, axis=0)
    sb = fb_stopbands(bloch, problem, problem.gammas, injected, bloch_config)
    edge = sb["edge"]
    diag: dict[str, Any] = {}
    out = bloch_fb_segmented_cmes_solve(
        x,
        bloch["seg_bounds"],
        y0_fwd,
        y0_bwd,
        np.transpose(sb["gammas"], (0, 2, 1)),
        problem.relations_3wm,
        problem.relations_4wm,
        bloch["base_3wm"],
        bloch["base_4wm"],
        bloch["prefac"],
        bloch["phi_plus"],
        bloch["phi_minus"],
        bloch["L_nl"],
        bloch["positions"],
        bloch["n_cells"],
        bloch["Zb_plus"],
        bloch["Zb_minus"],
        passes=bloch_config.passes,
        kappa_cutoff=bloch_config.kappa_cutoff,
        m_max=sb["m_max"],
        process_weights_3wm=sb["pw3"],
        process_weights_4wm=sb["pw4"],
        convergence_threshold=bloch_config.convergence_threshold,
        save_all_passes=bloch_config.save_all_passes,
        anderson_depth=bloch_config.anderson_depth,
        Ze=ports.Z_ext,
        Zout_in=Zout_in,
        Zout_out=Zout_out,
        abcd_in=abcd_in,
        abcd_out=abcd_out,
        edge_K=None if edge is None else edge["edge_K"],
        edge_mode=None if edge is None else edge["edge_mode"],
        diagnostics=diag,
        rk_config=rk_config,
    )
    bloch.update(diag)
    if len(problem.segments) == 1:
        bloch = _drop_segment_axis(bloch, problem.n_freq)

    s = problem.mode_labels.index(meta["signal_mode"])
    a_in, a_out = out["a_in"], out["a_out"]
    history: dict[str, Any] = {}
    if bloch_config.save_all_passes:
        history = {k: out[k] for k in ("I_fwd", "I_bwd", "b_in", "b_out")}
        history.update(
            port_gains(
                a_in[:, s], a_out[:, s], out["b_in"][..., s], out["b_out"][..., s]
            )
        )
    b_in, b_out = out["b_in"][-1], out["b_out"][-1]
    return GainResult(
        model=GainModel.BLOCH.value,
        x=x,
        mode_labels=problem.mode_labels,
        I_fwd=out["I_fwd"][-1],
        I_bwd=out["I_bwd"][-1],
        a_in=a_in,
        a_out=a_out,
        b_in=b_in,
        b_out=b_out,
        **port_gains(a_in[:, s], a_out[:, s], b_in[:, s], b_out[:, s]),
        converged=np.asarray(diag["converged"]),
        diagnostics=bloch,
        history=history,
        **meta,
    )
