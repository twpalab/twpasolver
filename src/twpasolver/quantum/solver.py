"""
Linearised quantum input-output model of a TWPA in the Bloch-mode basis.

Pipeline for each signal frequency:

1. classical background (pump, harmonics, reflections) and lossless Bloch tables
   (:mod:`twpasolver.quantum.background`);
2. kept linear terms and the flux-basis generator ``K(x)``, checked to satisfy
   ``K^dag Sigma + Sigma K = 0`` (:mod:`twpasolver.quantum.linearize`);
3. the device as a cascade of Gaussian channels: input edge, then per step
   ``loss(h/2) * parametric(h) * loss(h/2)``, then output edge, reduced with the
   Redheffer star product (:mod:`twpasolver.quantum.gaussian`);
4. observables (:mod:`twpasolver.quantum.observables`).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from twpasolver.cmes.solver_config import FBSolverConfig
from twpasolver.logger import log
from twpasolver.quantum.background import Background, build_background
from twpasolver.quantum.gaussian import (
    cascade,
    concat,
    gauss_nodes,
    loss_block,
    parametric_blocks,
    passive_block,
    star,
)
from twpasolver.quantum.kernels import (
    cascade_nb,
    collocation_weights,
    generator_nb,
    magnus_omega_nb,
    step_blocks_nb,
)
from twpasolver.quantum.linearize import (
    PumpField,
    active_weak_modes,
    generator,
    hamiltonian_defect,
    linear_terms,
)
from twpasolver.quantum.observables import bose, observables


def edge_matrices(bg: Background, i: int, active: list[int]):
    """
    Flux-normalised passive scattering of both device ends.

    Input end (left: external line, right: device): inputs ``(a, A^-(0))``,
    outputs ``(b, A^+(0))``. Output end (left: device, right: external line):
    inputs ``(A^+(L), a_out)``, outputs ``(A^-(L), b_out)``. ``a``/``b`` are the
    incident/reflected current waves of the external line,
    ``V = Ze (a + b)``; connector ABCD with port 2 facing the device.
    """
    M = len(active)
    Ze = bg.Ze
    P_in = np.zeros((2 * M, 2 * M), dtype=np.complex128)
    P_out = np.zeros((2 * M, 2 * M), dtype=np.complex128)
    for j, n in enumerate(active):
        w = bg.omegas[i, n]
        W = bg.W[i, n].real
        Zp, Zm = bg.Zb_plus[i, n], bg.Zb_minus[i, n]
        n_ext = np.sqrt(Ze / (2 * w))
        n_dev = np.sqrt(W / (4 * w))
        nrm = np.array([n_ext, n_dev])

        A, B, C, D = bg.abcd_in[i, n]
        Mm = np.array([[Ze, -(A * Zp + B)], [-1, -(C * Zp + D)]])
        Nn = np.array([[-Ze, -A * Zm + B], [-1, -C * Zm + D]])
        s = np.linalg.solve(Mm, Nn) * nrm[:, None] / nrm[None, :]  # (b, A+) <- (a, A-)
        P_in[j, j] = s[0, 0]
        P_in[j, M + j] = s[0, 1]
        P_in[M + j, j] = s[1, 0]
        P_in[M + j, M + j] = s[1, 1]

        A, B, C, D = bg.abcd_out[i, n]
        Mm = np.array([[Ze, A * Zm + B], [-1, C * Zm + D]])
        Nn = np.array([[-Ze, A * Zp - B], [-1, C * Zp - D]])
        s = (
            np.linalg.solve(Mm, Nn) * nrm[:, None] / nrm[None, :]
        )  # (b_o, A-) <- (a_o, A+)
        # The envelopes exclude the Bloch propagation: the boundary fields at
        # x = L are A^+ e^{-jkL} and A^- e^{+jkL}.
        ph = np.exp(-1j * bg.k[i, n] * bg.L)
        P_out[j, j] = s[1, 1] * ph * ph  # A-(L) <- A+(L)
        P_out[j, M + j] = s[1, 0] * ph  # A-(L) <- a_out
        P_out[M + j, j] = s[0, 1] * ph  # b_out <- A+(L)
        P_out[M + j, M + j] = s[0, 0]  # b_out <- a_out
    return P_in, P_out


def device_block(
    bg: Background,
    i: int,
    step_cells: float = 2.0,
    n_nodes: int = 4,
    temperature: float = 0.0,
    loss: bool = True,
    active: list[int] | None = None,
    check_commutators: bool = False,
    engine: str = "numba",
):
    """
    Device block ``(S, C, D)`` of frequency point ``i`` and diagnostics.

    Parameters
    ----------
    step_cells : float
        Step length in supercells. Kept terms oscillate by less than pi per
        supercell. The scheme keeps ``Omega_1 + Omega_2``; the neglected
        ``Omega_3`` grows as (coupling x step)^3, so the error grows ~ h^2:
        on the imskit holotype 2 supercells with 4 nodes is within 2e-4 dB of
        a converged reference (gain up to 36 dB, 70-ohm ports); 1 supercell
        with 2 nodes within 2e-5 dB.
    n_nodes : int
        Gauss collocation nodes per step (2 = classical 4th-order Magnus;
        use >= 4 for steps longer than one supercell).
    check_commutators : bool
        Propagate the noise commutator ``D`` through the cascade so that
        ``S J S^dag + D = J`` can be checked on the device (about 30% slower).
        Without it ``D`` is ``None``; the symplectic defect of the lossless
        steps is always reported (``step_defect``), and the loss, edge and star
        algebra preserve the identity exactly.
    engine : {"numba", "numpy"}
        Compiled kernels (default) or the reference numpy implementation.
    """
    if active is None:
        active = active_weak_modes(bg, i)
    terms = linear_terms(bg, i, active)
    M = len(active)
    d = bg.n_cells
    N = max(1, int(round(bg.L / (d * step_cells))))
    h = bg.L / N
    omegas = bg.omegas[i, active]
    nth = bose(omegas, temperature)
    alpha = bg.alpha[i, active] if loss else np.zeros(M)
    pf = PumpField(bg, i)
    P_in, P_out = edge_matrices(bg, i, active)
    edge_in = passive_block(P_in, nth)
    edge_out = passive_block(P_out, nth)
    ham = np.nan

    if engine == "numpy":
        x0 = np.arange(N) * h
        x1, x2 = gauss_nodes(x0, h)
        K1 = generator(terms, pf(x1), x1)
        K2 = generator(terms, pf(x2), x2)
        if check_commutators:
            ham = max(hamiltonian_defect(K1), hamiltonian_defect(K2))
        P = parametric_blocks(K1, K2, h)
        Lh = loss_block(np.exp(-alpha * h / 2), nth, N)
        steps = star(star(Lh, P), Lh)
        track: list = []
        S, D, C = (b[0] for b in cascade(concat(edge_in, steps, edge_out), track))
        diag = {"max_feedback": max(track) if track else 1.0, "step_defect": np.nan}
    else:
        c, w, Q = collocation_weights(n_nodes)
        x = (np.arange(N)[:, None] * h + c[None, :] * h).ravel()
        K = generator_nb(
            x,
            pf(x),
            terms.coeff,
            terms.phase,
            terms.slots,
            terms.tgt,
            terms.src,
            terms.scale,
            M,
        )
        if check_commutators:
            ham = hamiltonian_defect(K)
        Om = magnus_omega_nb(K, w, Q, h)
        t = np.exp(-alpha * h / 2)
        t4 = np.tile(t, 4)
        r2 = 1.0 - t4**2
        c4 = r2 * (np.tile(nth, 4) + 0.5)
        Jp = np.tile(np.r_[np.ones(M), -np.ones(M)], 2)
        S_st, C_st, D_st, defect = step_blocks_nb(
            Om, t4, c4, r2 * Jp, check_commutators
        )
        n = 4 * M
        if check_commutators:
            D_all = np.concatenate([edge_in[1], D_st, edge_out[1]])
        else:
            D_all = np.zeros((1, n, n), dtype=np.complex128)
        S, C, D, fb = cascade_nb(
            np.concatenate([edge_in[0], S_st, edge_out[0]]),
            np.concatenate([edge_in[2], C_st, edge_out[2]]),
            D_all,
            check_commutators,
        )
        if not check_commutators:
            D = None
        diag = {"max_feedback": float(fb), "step_defect": float(defect.max())}

    diag.update(
        {
            "n_terms": len(terms.coeff),
            "n_driven_rows": terms.n_driven,
            "hamiltonian_defect": ham,
            "n_steps": N,
            "h": h,
        }
    )
    return (S, C, D), terms, diag


def quantum_response(
    analysis,
    signal_freqs,
    pump: float | None = None,
    Ip0: float | None = None,
    mode_array_config: str = "basic_3wm",
    signal_mode: str = "s",
    pump_mode: str = "p",
    idler_mode: str | None = "i",
    thin: float | None = None,
    fb_config: FBSolverConfig | None = None,
    step_cells: float = 2.0,
    n_nodes: int = 4,
    temperature: float = 0.0,
    loss: bool = True,
    edge_guard: bool = True,
    background: Background | None = None,
    check_commutators: bool = False,
    engine: str = "numba",
) -> dict[str, Any]:
    """
    Linearised quantum response (gain, noise, efficiency, correlations).

    Parameters
    ----------
    analysis : TWPAnalysis
        Non-segmented device whose cells carry their loss as ``delta``.
    signal_freqs : array
        Signal frequencies (analysis unit).
    pump, Ip0, mode_array_config, signal_mode, pump_mode, idler_mode, thin,
    fb_config :
        As in ``TWPAnalysis.gain(model="bloch")``; ``fb_config`` controls the
        classical pump solve and the term selection (``kappa_cutoff``,
        ``m_max``).
    step_cells, n_nodes :
        Step length (supercells) and collocation nodes per step of the Magnus
        integrator; see :func:`device_block`.
    temperature : float
        Temperature (K) of the input ports and of the loss baths.
    loss : bool
        Include the loss channels (the pump keeps its classical loss).
    edge_guard : bool
        Apply the classical stopband-edge weights.
    background : Background, optional
        Reuse a precomputed classical background.
    check_commutators : bool
        Propagate the noise commutator through the cascade and report
        ``commutator_defect`` / ``commutator_signal`` (otherwise NaN; the
        per-step ``step_defect`` is always reported).
    engine : {"numba", "numpy"}
        Compiled kernels or the reference numpy implementation.

    Returns
    -------
    dict of arrays over the signal frequencies (NaN where the signal or idler is
    not a canonical travelling wave), plus per-point ``details``.
    """
    bg = background or build_background(
        analysis,
        signal_freqs,
        pump=pump,
        Ip0=Ip0,
        mode_array_config=mode_array_config,
        signal_mode=signal_mode,
        pump_mode=pump_mode,
        idler_mode=idler_mode,
        thin=thin,
        fb_config=fb_config,
        edge_guard=edge_guard,
    )
    F = len(bg.signal_freqs)
    keys = [
        "gain_db",
        "added_noise",
        "caves_bound",
        "quantum_efficiency",
        "qe_ideal",
        "qe_normalized",
        "commutator_signal",
        "commutator_defect",
        "squeezing_db",
        "antisqueezing_db",
        "epr_variance_db",
        "log_negativity",
        "idler_gain_db",
        "step_defect",
        "max_feedback",
    ]
    res: dict[str, Any] = {k: np.full(F, np.nan) for k in keys}
    for k in ("S21", "S11", "S22", "S12"):
        res[k] = np.full(F, np.nan, dtype=np.complex128)
    res["details"] = []
    sig = bg.mode_labels.index(bg.signal_mode)
    idl = bg.mode_labels.index(bg.idler_mode) if bg.idler_mode else None

    for i in range(F):
        active = active_weak_modes(bg, i)
        f_s, f_i = bg.freqs_signed[i, sig], (
            bg.freqs_signed[i, idl] if idl is not None else None
        )
        degenerate = f_i is not None and np.isclose(abs(f_s), abs(f_i), rtol=1e-9)
        if sig not in active or degenerate:
            res["details"].append({"skipped": "signal inactive or degenerate"})
            continue
        (S, C, D), terms, diag = device_block(
            bg,
            i,
            step_cells,
            n_nodes,
            temperature,
            loss,
            active,
            check_commutators,
            engine,
        )
        labels = [bg.mode_labels[n] for n in active]
        n_in = bose(bg.omegas[i, active], temperature)
        obs = observables(
            S,
            D,
            C,
            labels,
            n_in,
            signal=active.index(sig),
            idler=active.index(idl) if idl in active else None,
        )
        for k in keys + ["S21", "S11", "S22", "S12"]:
            if k in obs:
                res[k][i] = obs[k]
        res["step_defect"][i] = diag["step_defect"]
        res["max_feedback"][i] = diag["max_feedback"]
        res["details"].append(
            {
                **diag,
                "active_modes": labels,
                "noise_budget": obs["noise_budget"],
                "covariance_out": obs["covariance_out"],
                "quadrature_cov_si": obs.get("quadrature_cov_si"),
                "S": S,
                "D": D,
                "C": C,
            }
        )
        if diag["step_defect"] > 1e-10:
            log.warning(
                "Quantum Bloch model: lossless step not symplectic at %.4g "
                "(defect %.1e); reduce step_cells.",
                bg.signal_freqs[i],
                diag["step_defect"],
            )
        if diag["hamiltonian_defect"] > 1e-10:
            log.warning(
                "Quantum Bloch model: generator not Hamiltonian at %.4g (defect %.1e); "
                "check the term selection.",
                bg.signal_freqs[i],
                diag["hamiltonian_defect"],
            )
    res.update(
        {
            "signal_freqs": bg.signal_freqs,
            "pump_freq": bg.pump_freq,
            "mode_labels": bg.mode_labels,
            "weak_modes": [bg.mode_labels[n] for n in bg.weak],
            "background": bg,
        }
    )
    return res
