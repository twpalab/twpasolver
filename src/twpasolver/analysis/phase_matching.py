"""Phase mismatch of 3WM and 4WM processes over pump and signal frequencies."""

from typing import Any, Callable, Literal

import numpy as np

from twpasolver.analysis.results import PhaseMatchingResult
from twpasolver.cmes.prep import CMEProblem


def phase_matching(
    problem_at: Callable[[float], CMEProblem],
    signal_f: np.ndarray,
    pump_f: np.ndarray,
    stopbands: np.ndarray,
    seg_N_tot: np.ndarray,
    seg_xi: np.ndarray,
    Ip0: float,
    process: Literal["PA", "FCU", "FCD"],
    mixing: Literal["3WM", "4WM"],
    signal_mode: str,
    pump_mode: str,
    idler_mode: str,
    relative: bool,
    unit: str,
) -> PhaseMatchingResult:
    """
    Mismatch for every pump of ``pump_f``; ``problem_at(pump)`` gives the CME
    problem with the signal swept over ``signal_f``.

    Segmented devices accumulate ``sum_s N_s dk_s``, so ``delta`` is the
    length-weighted mean of the per-segment mismatch (own k, reflection and
    ``chi = Ip0**2 xi_s / 8``).
    """
    weights = seg_N_tot / seg_N_tot.sum()
    chi = Ip0**2 * seg_xi / 8
    idler_sign, signal_sign = {"PA": (-1, -1), "FCD": (1, -1), "FCU": (-1, 1)}[process]
    pump_coeff = 2 if mixing == "4WM" else 1

    deltas = np.full((len(signal_f), len(pump_f)), np.nan)
    f_triplets, k_triplets = [], []
    for i, p_freq in enumerate(pump_f):
        pr = problem_at(p_freq)
        ip, is_, ii = (
            pr.mode_labels.index(m) for m in (pump_mode, signal_mode, idler_mode)
        )
        k = -pr.gammas.imag  # (n_signal, n_modes, n_seg)
        kp = k[0, ip][:, None]
        g2 = np.abs(pr.reflections[0, ip])[:, None] ** 2
        ks, ki = k[:, is_].T, k[:, ii].T

        dk = (
            pump_coeff * kp
            + signal_sign * ks
            + idler_sign * ki
            + (chi[:, None] * (1 + g2))
            * (pump_coeff * kp + 2 * signal_sign * ks + 2 * idler_sign * ki)
        )
        deltas[:, i] = weights @ dk
        ks_mean, ki_mean = weights @ ks, weights @ ki
        if relative:
            deltas[:, i] /= ks_mean
        kp_mean = (weights @ kp).item()
        for j in range(len(signal_f)):
            f_triplets.append([p_freq, pr.freqs[j, is_], pr.freqs[j, ii]])
            k_triplets.append([kp_mean, ks_mean[j], ki_mean[j]])

    # A pump inside a stopband does not propagate: no phase matching there.
    in_gap = (
        (pump_f[:, None] >= stopbands[:, 0]) & (pump_f[:, None] <= stopbands[:, 1])
    ).any(axis=1)
    deltas[:, in_gap] = np.nan

    valid = ~np.isnan(deltas).all(axis=0)
    mean_delta = np.full(len(pump_f), np.inf)
    mean_delta[valid] = np.nanmean(np.abs(deltas[:, valid]), axis=0)
    meta: dict[str, Any] = dict(process=process, mixing=mixing, relative=relative)
    return PhaseMatchingResult(
        unit=unit,
        signal_mode=signal_mode,
        pump_mode=pump_mode,
        idler_mode=idler_mode,
        signal_freqs=signal_f,
        pump_freqs=pump_f,
        delta=deltas,
        optimal_pump_freq=(
            float(pump_f[np.argmin(mean_delta)]) if valid.any() else np.nan
        ),
        f_triplets=np.array(f_triplets).reshape(-1, 3),
        k_triplets=np.array(k_triplets).reshape(-1, 3),
        **meta,
    )
