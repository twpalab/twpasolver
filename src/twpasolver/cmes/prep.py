"""CME input preparation: relation coefficients, per-segment Bloch tables, stopband treatment."""

from dataclasses import dataclass
from typing import Any, Optional, Tuple

import numpy as np

from twpasolver.cmes import solver_config
from twpasolver.cmes.bloch import (
    bloch_mode_tables,
    edge_basis_tables,
    edge_reference_frequency,
    in_stopband,
    lossless_copy,
    process_weights,
    reactive_angle_deg,
    stopband_edges,
)
from twpasolver.cmes.solver_config import BlochSolverConfig
from twpasolver.logger import log


@dataclass
class CMEProblem:
    """
    Inputs of the CME solvers for one set of mode frequencies; a uniform device
    is one segment.

    Attributes:
        mode_labels: Mode names, in the order of the amplitudes.
        freqs: Signed mode frequencies in the analysis unit, (n_freq, n_modes).
        freqs_hz: ``|freqs|`` in Hz.
        gammas: Kernel propagation constants ``-alpha - j k`` per unit cell,
            (n_freq, n_modes, n_seg).
        reflections: Reflection coefficient against ``twpa.Z0_ref``,
            (n_freq, n_modes, n_seg).
        relations_3wm, relations_4wm: Mode-index relations, (n_rel, 3) / (n_rel, 4).
        unit_3wm, unit_4wm: Relation coefficients for unit epsilon / xi, (n_rel,).
        coeffs_3wm, coeffs_4wm: Coefficients of every segment, (n_rel, n_seg).
        segments: Uniform segments (``TWPA``), each a whole number of supercells.
        seg_bounds, seg_midpoints: Segment boundaries and midpoints (unit cells).
    """

    mode_labels: list[str]
    freqs: np.ndarray
    freqs_hz: np.ndarray
    gammas: np.ndarray
    reflections: np.ndarray
    relations_3wm: np.ndarray
    relations_4wm: np.ndarray
    unit_3wm: np.ndarray
    unit_4wm: np.ndarray
    coeffs_3wm: np.ndarray
    coeffs_4wm: np.ndarray
    segments: list
    seg_bounds: np.ndarray
    seg_midpoints: np.ndarray

    @property
    def n_freq(self) -> int:
        """Number of frequency points."""
        return self.freqs.shape[0]

    @property
    def n_modes(self) -> int:
        """Number of modes."""
        return len(self.mode_labels)


def prepare_relations_coefficients(terms_3wm, terms_4wm, epsilon, xi):
    """
    Mode-index relations and coefficients of the 3WM and 4WM terms.

    Parameters
    ----------
    terms_3wm, terms_4wm : list
        RWA terms ``(mode, (indices...), coefficient)`` from
        ``ModeSet.rwa_terms``.
    epsilon, xi : float
        Nonlinearities; the coefficients are scaled by ``epsilon / 4`` and
        ``xi / 4``. An empty term list gives a single dummy zero relation.

    Returns
    -------
    relations_3wm : np.ndarray, int64, (n_rel, 3)
    relations_4wm : np.ndarray, int64, (n_rel, 4)
    coeffs_3wm    : np.ndarray, complex128, (n_rel,)
    coeffs_4wm    : np.ndarray, complex128, (n_rel,)
    """

    def extract_indices(terms):
        return [[term[0], *term[1]] for term in terms]

    def calculate_coefficients(terms, scaling_factor):
        return np.array([term[-1] * scaling_factor for term in terms])

    relations_3wm = extract_indices(terms_3wm) if terms_3wm else [[0, 0, 0]]
    relations_4wm = extract_indices(terms_4wm) if terms_4wm else [[0, 0, 0, 0]]
    coeffs_3wm = (
        calculate_coefficients(terms_3wm, epsilon / 4) if terms_3wm else np.array([0])
    )
    coeffs_4wm = (
        calculate_coefficients(terms_4wm, xi / 4) if terms_4wm else np.array([0])
    )

    return (
        np.array(relations_3wm, dtype=np.int64),
        np.array(relations_4wm, dtype=np.int64),
        np.array(coeffs_3wm, dtype=np.complex128),
        np.array(coeffs_4wm, dtype=np.complex128),
    )


def _harmonic_limits(
    split: np.ndarray,
    W: np.ndarray,
    kd_pi: np.ndarray,
    base: int,
    e_edge: float,
) -> np.ndarray:
    """
    Per-mode harmonic limits of ``BlochSolverConfig.m_max='auto'``.

    ``base`` for every mode, raised to the Bragg order ``m = round(k d / pi)``
    of a mode inside a stopband (``|arg W| > 45 deg``) or close to its edge
    (``split < e_edge``, twice ``edge_threshold``): its counter-propagating component, at
    ``-m pi / d``, is reached only through ``G_m``. The zone centre
    (``k d / pi < 0.5``) is excluded. All arrays share the shape
    ``(..., n_modes)``.
    """
    zone_centre = kd_pi < 0.5
    special = ((split < e_edge) | (reactive_angle_deg(W) > 45.0)) & ~zone_centre
    order = np.round(kd_pi).astype(np.int64)
    return np.where(special, np.maximum(base, order), base).astype(np.int64)


def _stopband_guard_weights(
    split: np.ndarray,
    W: np.ndarray,
    kd_pi: np.ndarray,
    e_edge: float,
) -> np.ndarray:
    """
    Per-mode weights of ``BlochSolverConfig.simulate_stopband_modes=False``.

    1 for ``split >= e_edge``, smoothstep down to 0 at the edge on the passband
    side, and 0 inside a stopband (``|arg W| > 45 deg``). The zone centre
    (``k d / pi < 0.5``), where the splitting also vanishes but the Bloch basis
    stays valid, keeps weight 1. All arrays share the shape ``(..., n_modes)``.
    """
    zone_centre = kd_pi < 0.5
    t = np.clip(np.asarray(split, dtype=np.float64) / e_edge, 0.0, 1.0)
    w = t * t * (3.0 - 2.0 * t)
    w = np.where(reactive_angle_deg(W) > 45.0, 0.0, w)
    return np.where(zone_centre, 1.0, w)


def _edge_basis_entries(
    device: Any,
    split: np.ndarray,
    kappa: np.ndarray,
    freqs_hz: np.ndarray,
    n_cells: int,
    injected: np.ndarray,
    e_edge: float,
    in_gap: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, Optional[dict[str, Any]]]:
    """
    Modes of one uniform device (or segment) to expand on the fixed edge basis.

    Parameters
    ----------
    device : TWPA
        Uniform device or segment; its supercell defines the Bloch modes.
    split, kappa, freqs_hz : (n_freq, n_modes)
        Bloch splitting ``|sinh(gamma d)|``, kernel wavenumber (``Im`` of the
        kernel gamma, per unit cell) and mode frequency (Hz).
    n_cells : int
        Unit cells per supercell.
    injected : (n_modes,) bool
        Modes with an input, never flagged.
    e_edge : float
        Threshold on the splitting, also the anchor of the fixed basis.
    in_gap : (n_freq, n_modes) bool, optional
        Entries inside a stopband to flag as well, whatever their splitting
        (``BlochSolverConfig.simulate_stopband_modes``).

    Returns
    -------
    near_injected : (n_modes,) bool
        Injected modes that lie below the threshold somewhere.
    flag : (n_freq, n_modes) bool
        Entries expanded on the fixed basis.
    entries : dict or None
        None when nothing is flagged. Otherwise ``ii, nn`` (indices of the
        flagged entries, empty when none has an edge), ``f_mode``, ``f_ref``,
        ``tab`` (:func:`edge_basis_tables`), ``missing`` (modes with no
        stopband edge found) and ``outside`` (modes below the threshold outside
        any stopband and edge window); both kept in the native basis.

    An entry below the threshold is expanded on the fixed basis only if it lies
    inside a stopband or between the nearest edge and its reference frequency,
    the window where the splitting of that edge stays below ``e_edge``. The
    splitting also vanishes where two bands touch without a gap (zone folding
    of a supercell of repeated periods); the native basis is exact there, while
    a fixed basis anchored at a distant edge would carry the wrong carrier
    wavenumber.
    """
    # The splitting also vanishes at the zone centre (k -> 0, low frequency),
    # where T -> I and the two Bloch modes stay distinct (W finite): no
    # stopband edge there, the native basis is exact.
    zone_centre = np.abs(kappa) * n_cells / np.pi < 0.5
    near = (split < e_edge) & ~zone_centre
    if in_gap is not None:
        near |= in_gap & ~zone_centre
    near_injected = injected & near.any(axis=0)
    flag = near & ~injected[None, :]
    if not flag.any():
        return near_injected, flag, None
    twpa_ll = lossless_copy(device)
    ii, nn = np.nonzero(flag)
    f_mode = freqs_hz[ii, nn]
    # Edge index: k d = m pi at the edge, m with the parity of the edge.
    kd_pi = np.abs(kappa[ii, nn]) * n_cells / np.pi
    m_edge = np.zeros(len(f_mode), dtype=np.int64)
    f_ref = np.empty(len(f_mode))
    f_edge = np.full(len(f_mode), np.nan)
    found = np.ones(len(f_mode), dtype=bool)
    # Every edge near the flagged frequencies, found in one scan; each entry
    # takes the nearest one. An edge farther than the scanned span may not be
    # the nearest, so those entries are rescanned with a wider span, and the
    # per-frequency search is the last resort.
    par_q = np.zeros(len(f_mode), dtype=np.int64)
    todo = np.ones(len(f_mode), dtype=bool)
    gap_entry = in_stopband(twpa_ll, f_mode)
    outside = np.zeros(len(f_mode), dtype=bool)
    for span in (1.0e9, 4.0e9, 8.0e9):
        if not todo.any():
            break
        e_tab, ref_tab, par_tab = stopband_edges(twpa_ll, f_mode[todo], e_edge, span)
        if not len(e_tab):
            continue
        idx = np.flatnonzero(todo)
        near = np.argmin(np.abs(e_tab[None, :] - f_mode[idx, None]), axis=1)
        ok = (np.abs(e_tab[near] - f_mode[idx]) <= span) & np.isfinite(ref_tab[near])
        f_ref[idx[ok]] = ref_tab[near[ok]]
        f_edge[idx[ok]] = e_tab[near[ok]]
        par_q[idx[ok]] = par_tab[near[ok]]
        todo[idx[ok]] = False
    for q, f in enumerate(f_mode):
        if not todo[q]:
            parity = int(par_q[q])
        else:
            try:
                f_ref[q], f_edge[q], parity = edge_reference_frequency(
                    twpa_ll, f, e_edge
                )
            except ValueError:
                # No stopband edge around f (e.g. a closed gap): no
                # exceptional point, the native basis is kept.
                found[q] = False
                continue
        # Inside a stopband, or inside the switch-over window [f_edge, f_ref]
        # of the edge; otherwise the small splitting is a band touching
        # without a gap and the native basis is kept.
        lo, hi = sorted((f_edge[q], f_ref[q]))
        if not (gap_entry[q] or lo <= f <= hi):
            outside[q] = True
            continue
        odd = 0 if parity > 0 else 1
        m_edge[q] = 2 * int(np.round((kd_pi[q] - odd) / 2)) + odd
        if abs(kd_pi[q] - m_edge[q]) > 0.5:
            log.warning(
                "Edge basis: a mode at %.4f GHz has k d / pi = %.3f, far from its "
                "edge index %d.",
                f / 1e9,
                kd_pi[q],
                m_edge[q],
            )
    missing = np.unique(nn[~found])
    outside_modes = np.unique(nn[outside])
    found &= ~outside
    if not found.all():
        flag[ii[~found], nn[~found]] = False
        ii, nn, f_mode = ii[found], nn[found], f_mode[found]
        f_ref, m_edge = f_ref[found], m_edge[found]
    tab = (
        edge_basis_tables(device, twpa_ll, f_mode, f_ref, m_edge)
        if found.any()
        else None
    )
    entries = {
        "ii": ii,
        "nn": nn,
        "f_mode": f_mode,
        "f_ref": f_ref,
        "tab": tab,
        "missing": missing,
        "outside": outside_modes,
    }
    return near_injected, flag, entries


def _log_edge_basis_skipped(labels, near_injected, entries, e_edge):
    """Warnings of the edge basis: injected modes near an edge, edges not found."""
    if near_injected.any():
        log.warning(
            "Edge basis: mode(s) %s with an input lie close to a stopband edge "
            "and keep their native Bloch basis; their description there is "
            "unreliable.",
            [labels[j] for j in np.flatnonzero(near_injected)],
        )
    if entries is not None and len(entries["missing"]):
        log.warning(
            "Edge basis: no stopband edge found near mode(s) %s with "
            "|sinh(gamma d)| < %g at some points; they keep their native "
            "Bloch basis there.",
            [labels[j] for j in entries["missing"]],
            e_edge,
        )
    if entries is not None and len(entries.get("outside", ())):
        log.info(
            "Edge basis: mode(s) %s have |sinh(gamma d)| < %g at some points "
            "outside any stopband and edge window (bands touching without a "
            "gap); they keep their native Bloch basis there.",
            [labels[j] for j in entries["outside"]],
            e_edge,
        )


def fb_stopbands(
    bloch: dict[str, Any],
    problem: CMEProblem,
    gammas: np.ndarray,
    injected: np.ndarray,
    bloch_config: BlochSolverConfig,
) -> dict[str, Any]:
    """
    Stopband treatment and harmonic limits of the Bloch forward-backward models.

    ``bloch_config.edge_threshold`` marks the modes close to an edge;
    ``bloch_config.simulate_stopband_modes`` either expands them, and every
    mode inside a stopband, on the fixed edge basis
    (:func:`apply_edge_basis_segmented`), or switches their processes off
    with :func:`_stopband_guard_weights`. Modes with an input are never
    changed. ``m_max='auto'`` gives per-mode harmonic limits
    (:func:`_harmonic_limits`, window ``2 * edge_threshold``).

    Returns a dict with the kernel ``gammas``, ``edge`` (edge-basis tables or
    None), ``m_max`` (int or per-mode table) and the process weights ``pw3``,
    ``pw4`` (None when every process has weight 1).
    """
    e = bloch_config.edge_threshold
    labels = problem.mode_labels
    kd_pi = (
        np.abs(np.transpose(gammas.imag, (0, 2, 1)))
        * np.asarray(bloch["n_cells"])[None, :, None]
        / np.pi
    )
    if bloch_config.m_max == "auto":
        m_max = _harmonic_limits(
            bloch["split"],
            bloch["W"],
            kd_pi,
            solver_config.BLOCH_M_MAX_AUTO_BASE,
            2.0 * (e if e is not None else 0.10),
        )
        bloch["m_lim"] = m_max
    else:
        m_max = np.asarray(bloch_config.m_max)
    out: dict[str, Any] = dict(
        gammas=gammas, edge=None, m_max=m_max, pw3=None, pw4=None
    )
    if e is None:
        return out
    if bloch_config.simulate_stopband_modes:
        edge = apply_edge_basis_segmented(bloch, problem, injected, e)
        if edge is not None:
            out["edge"] = edge
            out["gammas"] = edge["gammas"]  # kernel and carrier: -1j kappa_e
        return out
    weights = _stopband_guard_weights(bloch["split"], bloch["W"], kd_pi, e)
    lead = tuple(range(weights.ndim - 1))
    near_inj = injected & (weights < 1.0).any(axis=lead)
    if near_inj.any():
        log.warning(
            "Stopband guard: mode(s) %s with an input lie in a stopband or close "
            "to its edge and are kept in the native Bloch basis; their "
            "description there is unreliable.",
            [labels[j] for j in np.flatnonzero(near_inj)],
        )
    weights[..., injected] = 1.0
    bloch["edge_weights"] = weights
    if np.all(weights == 1.0):
        return out
    out["pw3"] = process_weights(problem.relations_3wm, weights)
    out["pw4"] = process_weights(problem.relations_4wm, weights)
    return out


def apply_edge_basis_segmented(
    bloch: dict[str, Any],
    problem: CMEProblem,
    injected: np.ndarray,
    threshold: float,
) -> Optional[dict[str, Any]]:
    """
    Fixed ("edge") Bloch basis for modes close to a stopband edge.

    A mode without input whose splitting in segment ``s`` is below
    ``threshold``, or that lies inside a stopband, is expanded there on the
    lossless Bloch pair of that segment's supercell, anchored on the
    passband side of the nearest edge (:func:`_edge_basis_entries`).
    The junction coefficients follow from the overwritten ``Zb_plus`` and
    ``Zb_minus`` and stay exact. A mode flagged in at least one segment
    (``edge_mode``) is solved exactly along the whole line; in its native
    segments the generator is ``diag(-alpha, alpha)``.

    Returns None when nothing is flagged, otherwise the kernel gammas
    (``(n_freq, n_modes, n_seg)``), ``edge_K`` (``(n_freq, n_seg, n_modes,
    2, 2)``, per unit cell), ``edge_flag`` (``(n_freq, n_seg, n_modes)``) and
    ``edge_mode`` (``(n_freq, n_modes)``); diagnostics are stored in
    ``bloch``.
    """
    e_edge = float(threshold)
    gammas = problem.gammas  # (n_freq, n_modes, n_seg)
    labels = problem.mode_labels
    segments = problem.segments
    n_freq, n_seg, n_modes = bloch["split"].shape
    freqs_hz = bloch["freqs_hz"]
    flag = np.zeros((n_freq, n_seg, n_modes), dtype=bool)
    near_inj = np.zeros(n_modes, dtype=bool)
    missing: set[int] = set()
    outside: set[int] = set()
    gammas_k = gammas.copy()
    edge_K = np.zeros((n_freq, n_seg, n_modes, 2, 2), dtype=np.complex128)
    f_ref_tab = np.full((n_freq, n_seg, n_modes), np.nan)
    ham = np.zeros((n_freq, n_seg, n_modes))
    cache: dict[Any, Any] = {}
    for s, seg in enumerate(segments):
        split_s = bloch["split"][:, s]
        kappa_s = gammas[:, :, s].imag
        in_gap = reactive_angle_deg(bloch["W"][:, s]) > 45.0
        key = (
            repr([c.model_dump() for c in seg.cells]),
            np.ascontiguousarray(kappa_s).tobytes(),
            np.ascontiguousarray(split_s).tobytes(),
        )
        if key not in cache:
            cache[key] = _edge_basis_entries(
                seg,
                split_s,
                kappa_s,
                freqs_hz,
                int(bloch["n_cells"][s]),
                injected,
                e_edge,
                in_gap=in_gap,
            )
        ni, fl, ent = cache[key]
        near_inj |= ni
        flag[:, s] = fl
        if ent is None:
            continue
        missing.update(int(j) for j in ent["missing"])
        outside.update(int(j) for j in ent["outside"])
        if len(ent["ii"]) == 0:
            continue
        ii, nn, tab = ent["ii"], ent["nn"], ent["tab"]
        n_nl = tab["phi_plus"].shape[-1]
        bloch["Zb_plus"][ii, s, nn] = tab["Zb_plus"]
        bloch["Zb_minus"][ii, s, nn] = tab["Zb_minus"]
        bloch["W"][ii, s, nn] = tab["W"]
        bloch["phi_plus"][ii, s, nn, :n_nl] = tab["phi_plus"]
        bloch["phi_minus"][ii, s, nn, :n_nl] = tab["phi_minus"]
        gammas_k[ii, nn, s] = -1j * tab["kappa_e"]
        edge_K[ii, s, nn] = tab["K"]
        f_ref_tab[ii, s, nn] = ent["f_ref"]
        ham[ii, s, nn] = tab["ham_defect"]
    _log_edge_basis_skipped(
        labels,
        near_inj,
        {
            "missing": np.array(sorted(missing), int),
            "outside": np.array(sorted(outside), int),
        },
        e_edge,
    )
    if not flag.any():
        return None
    edge_mode = np.any(flag, axis=1, keepdims=False).astype(bool)
    # Native segments of a mode solved exactly: decoupled directions with
    # the kernel attenuation (forward decays along +x, backward along -x).
    for i, n in zip(*np.nonzero(edge_mode)):
        for s in np.flatnonzero(~flag[i, :, n]):
            a = gammas_k[i, n, s].real
            edge_K[i, s, n] = np.array([[a, 0.0], [0.0, -a]], dtype=np.complex128)
    omegas = 2 * np.pi * freqs_hz
    bloch["prefac"] = -2j * omegas[:, None, :] / bloch["W"]
    seg_len = np.diff(bloch["seg_bounds"])
    bragg = np.abs(edge_K[..., 0, 1]) * seg_len[None, :, None]
    bloch.update(
        edge_flag=flag,
        edge_mode=edge_mode,
        edge_f_ref=f_ref_tab,
        edge_bragg_strength=bragg,
        edge_ham_defect=ham,
    )
    for n in np.flatnonzero(edge_mode.any(axis=0)):
        log.info(
            "Edge basis: mode %s at %d points, flagged in %d of %d segments, "
            "max Bragg strength |K01| L_seg = %.1f.",
            labels[n],
            int(edge_mode[:, n].sum()),
            int(flag[:, :, n].any(axis=0).sum()),
            n_seg,
            float(bragg[:, :, n].max()),
        )
    return {
        "gammas": gammas_k,
        "edge_K": edge_K,
        "edge_flag": flag,
        "edge_mode": edge_mode,
    }


def prepare_bloch_segmented(problem: CMEProblem) -> dict[str, Any]:
    """
    Per-segment Bloch tables for the forward-backward solver.

    Every segment is a uniform ``TWPA`` of whole supercells; its Bloch modes come
    from :func:`bloch_mode_tables` on its own supercell, with the branch of ``k``
    set by its kernel wavenumber. Intracell arrays are zero-padded to the largest
    number of nonlinear inductors per supercell.

    Returns
    -------
    dict
        ``W``, ``split``, ``Zb_plus``, ``Zb_minus``, ``prefac`` :
        (n_freq, n_seg, n_modes)
        ``phi_plus``, ``phi_minus`` : (n_freq, n_seg, n_modes, n_nl)
        ``L_nl``, ``positions`` : (n_seg, n_nl); ``n_cells`` : (n_seg,)
        ``base_3wm``, ``base_4wm`` : (n_seg, n_rel)
        ``seg_bounds`` : (n_seg + 1,) and the diagnostics ``eig_residual``,
        ``gamma_mismatch`` : (n_freq, n_seg, n_modes).
    """
    n_freq, n_modes = problem.n_freq, problem.n_modes
    gammas = problem.gammas  # (n_freq, n_modes, n_seg)
    segments, seg_bounds = problem.segments, problem.seg_bounds
    n_seg = len(segments)
    unit_3wm, unit_4wm = problem.unit_3wm, problem.unit_4wm
    freqs_hz = problem.freqs_hz
    omegas = 2 * np.pi * freqs_hz

    cache: dict[Any, dict[str, Any]] = {}
    per_seg = []
    for s, seg in enumerate(segments):
        g_ref = -gammas[:, :, s]
        key = (
            repr([c.model_dump() for c in seg.cells]),
            np.ascontiguousarray(g_ref).tobytes(),
        )
        if key not in cache:
            cache[key] = bloch_mode_tables(seg, freqs_hz.ravel(), g_ref.ravel())
        tab = cache[key]
        if seg.N_tot != seg.N * tab["n_cells"]:
            raise ValueError(f"Segment {s} does not hold a whole number of supercells.")
        per_seg.append(tab)
    log.info("Bloch tables computed for %d distinct of %d segments.", len(cache), n_seg)

    n_nl = max(t["L_nl"].size for t in per_seg)
    W = np.empty((n_freq, n_seg, n_modes), dtype=np.complex128)
    Zb_plus = np.empty_like(W)
    Zb_minus = np.empty_like(W)
    eig_res = np.empty((n_freq, n_seg, n_modes))
    split = np.empty((n_freq, n_seg, n_modes))
    g_mis = np.empty((n_freq, n_seg, n_modes))
    phi_plus = np.zeros((n_freq, n_seg, n_modes, n_nl), dtype=np.complex128)
    phi_minus = np.zeros_like(phi_plus)
    L_nl = np.zeros((n_seg, n_nl))
    positions = np.zeros((n_seg, n_nl))
    n_cells = np.empty(n_seg, dtype=np.int64)
    for s, tab in enumerate(per_seg):
        nn = tab["L_nl"].size
        W[:, s] = tab["W"].reshape(n_freq, n_modes)
        Zb_plus[:, s] = tab["Zb_plus"].reshape(n_freq, n_modes)
        Zb_minus[:, s] = tab["Zb_minus"].reshape(n_freq, n_modes)
        eig_res[:, s] = tab["eig_residual"].reshape(n_freq, n_modes)
        split[:, s] = tab["split"].reshape(n_freq, n_modes)
        g_mis[:, s] = np.abs(
            tab["gamma_exact"].reshape(n_freq, n_modes) + gammas[:, :, s]
        )
        phi_plus[:, s, :, :nn] = tab["phi_plus"].reshape(n_freq, n_modes, nn)
        phi_minus[:, s, :, :nn] = tab["phi_minus"].reshape(n_freq, n_modes, nn)
        L_nl[s, :nn] = tab["L_nl"]
        positions[s, :nn] = tab["positions"]
        n_cells[s] = tab["n_cells"]

    return {
        "W": W,
        "split": split,
        "Zb_plus": Zb_plus,
        "Zb_minus": Zb_minus,
        "phi_plus": phi_plus,
        "phi_minus": phi_minus,
        "L_nl": L_nl,
        "positions": positions,
        "n_cells": n_cells,
        "seg_bounds": seg_bounds,
        "eig_residual": eig_res,
        "gamma_mismatch": g_mis,
        "base_3wm": np.stack([unit_3wm * seg.epsilon for seg in segments]),
        "base_4wm": np.stack([unit_4wm * seg.xi for seg in segments]),
        "prefac": -2j * omegas[:, None, :] / W,
        "freqs_hz": freqs_hz,
    }
