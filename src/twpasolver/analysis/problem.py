"""One CME problem builder for both gain models and phase matching."""

from typing import Any, Mapping

import numpy as np

from twpasolver.analysis.linear import LinearData, ParameterInterpolator
from twpasolver.cmes.prep import CMEProblem, prepare_relations_coefficients
from twpasolver.modes_rwa import ModeSet


def build_problem(
    modes: ModeSet,
    tones: Mapping[str, Any],
    lin: LinearData,
    interp: ParameterInterpolator,
    segments: list,
    unit_multiplier: float,
) -> CMEProblem:
    """
    CME inputs for the mode frequencies set by the independent ``tones``.

    ``tones`` maps every independent mode to its frequency in the analysis unit
    (scalars or arrays of one length); mode parameters are interpolated at
    ``|f|`` from ``interp`` and relations follow from the RWA terms of ``modes``.
    """
    f = modes.frequencies(tones)
    freqs = np.ascontiguousarray(f.reshape(len(modes.labels), -1).T)
    n_freq, n_modes = freqs.shape
    lo, hi = lin.freqs[0], lin.freqs[-1]
    outside = (np.abs(freqs) < lo) | (np.abs(freqs) > hi)
    if outside.any():
        bad = sorted({modes.labels[j] for j in np.flatnonzero(outside.any(axis=0))})
        raise ValueError(
            f"Modes {bad} reach |f| = {np.abs(freqs)[outside].min():.3g} to "
            f"{np.abs(freqs)[outside].max():.3g}, outside the frequency grid [{lo:.3g}, "
            f"{hi:.3g}] of the linear response; extend TWPAnalysis.freqs to cover them."
        )
    n_seg = len(segments)
    p = interp.interpolate(np.abs(freqs).ravel())

    def table(key: str) -> np.ndarray:
        """(n_freq, n_modes, n_seg) from the (n_seg, n_freq * n_modes) interpolation."""
        return p[key].reshape(n_seg, n_freq, n_modes).transpose(1, 2, 0)

    # Relations are segment-independent; only epsilon and xi vary.
    rel3, rel4, unit3, unit4 = prepare_relations_coefficients(
        modes.rwa_terms(2), modes.rwa_terms(3), 1.0, 1.0
    )
    return CMEProblem(
        mode_labels=list(modes.labels),
        freqs=freqs,
        freqs_hz=np.abs(freqs) * unit_multiplier,
        gammas=-table("alpha") - 1j * table("k"),
        reflections=table("gamma"),
        relations_3wm=rel3,
        relations_4wm=rel4,
        unit_3wm=unit3,
        unit_4wm=unit4,
        coeffs_3wm=np.stack([unit3 * eps for eps in lin.seg_epsilon], axis=-1),
        coeffs_4wm=np.stack([unit4 * xi for xi in lin.seg_xi], axis=-1),
        segments=segments,
        seg_bounds=np.asarray(lin.seg_bounds, dtype=float),
        seg_midpoints=lin.seg_midpoints,
    )
