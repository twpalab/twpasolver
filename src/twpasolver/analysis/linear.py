"""Linear response of a TWPA: Bloch parameters per segment, stopbands, ``k*``."""

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.interpolate import CubicSpline

from twpasolver.cmes.bloch import bloch_wavenumber, in_stopband, lossless_copy
from twpasolver.logger import log
from twpasolver.models import TWPA
from twpasolver.twoport import TwoPortCell


@dataclass(frozen=True)
class LinearData:
    """
    Linear response of a TWPA on the analysis frequency grid.

    Frequencies are in the analysis unit. Per-segment arrays have shape
    ``(n_segments, n_freq)``; a uniform device is one segment.

    Attributes:
        freqs: Frequency grid, shape (n_freq,).
        abcd: ABCD matrices of the whole device, shape (n_freq, 2, 2).
        S21, S11: Scattering parameters of the whole device against ``twpa.Z0_ref``.
        stopbands: ``[start, end]`` of every stopband of the (first segment's)
            lossless supercell in the grid, shape (n_bands, 2).
        k, alpha: Bloch wavenumber and attenuation per unit cell.
        Zb: Bloch impedance.
        reflection_coeff: Reflection coefficient of ``Zb`` against ``twpa.Z0_ref``.
        k_star: Length-weighted mean ``k`` minus the linear part of a quadratic fit
            below a tenth of the first stopband (the bare ``k`` without stopband),
            shape (n_freq,).
        seg_bounds: Segment boundaries in unit cells, shape (n_segments + 1,).
        seg_N_tot: Unit cells per segment.
        seg_epsilon, seg_xi: Nonlinearity coefficients per segment.
    """

    freqs: np.ndarray
    abcd: np.ndarray
    S21: np.ndarray
    S11: np.ndarray
    stopbands: np.ndarray
    k: np.ndarray
    alpha: np.ndarray
    Zb: np.ndarray
    reflection_coeff: np.ndarray
    k_star: np.ndarray
    seg_bounds: np.ndarray
    seg_N_tot: np.ndarray
    seg_epsilon: np.ndarray
    seg_xi: np.ndarray

    @property
    def S21_db(self) -> np.ndarray:
        """``|S21|`` in dB."""
        return 20 * np.log10(np.abs(self.S21))

    @property
    def seg_midpoints(self) -> np.ndarray:
        """Segment midpoints in unit cells."""
        return 0.5 * (self.seg_bounds[:-1] + self.seg_bounds[1:])


class ParameterInterpolator:
    """
    Cubic-spline interpolation over frequency of 1-D ``(n_freq,)`` or 2-D
    ``(n, n_freq)`` arrays (row by row), real or complex; frequencies sorted.
    """

    def __init__(self, frequencies: np.ndarray, params: dict[str, np.ndarray]):
        """Spline every parameter of ``params`` over ``frequencies``."""
        self.frequencies = np.asarray(frequencies)
        self._splines: dict[str, tuple[Any, Any]] = {}
        for name, values in params.items():
            values = np.asarray(values)

            def spline(arr: np.ndarray) -> CubicSpline:
                return CubicSpline(self.frequencies, arr, axis=-1)

            self._splines[name] = (
                spline(np.real(values)),
                spline(np.imag(values)) if np.iscomplexobj(values) else None,
            )

    def interpolate(self, frequencies: Any) -> dict[str, np.ndarray]:
        """Every parameter at ``frequencies``; 2-D parameters give ``(n, len(frequencies))``."""
        freqs = np.atleast_1d(np.asarray(frequencies, dtype=float))
        out = {}
        for name, (re, im) in self._splines.items():
            out[name] = re(freqs) if im is None else re(freqs) + 1j * im(freqs)
        return out


def is_segmented(twpa: TWPA) -> bool:
    """Whether ``twpa`` is a chain of ``TWPA`` segments."""
    return bool(twpa.cells) and all(isinstance(cell, TWPA) for cell in twpa.cells)


def segments_of(twpa: TWPA) -> list[TWPA]:
    """Uniform segments of ``twpa``; a uniform device is one segment."""
    return list(twpa.cells) if is_segmented(twpa) else [twpa]


def _bloch_params(cell: TWPA, f: np.ndarray, Z0: Any, T: Any) -> dict[str, Any]:
    """
    Bloch parameters per unit cell of a uniform segment with supercell ABCD ``T``.

    ``k`` (winding rule, :func:`bloch_wavenumber`), ``alpha``, the Bloch impedance
    ``Zb`` and its ``reflection_coeff`` against ``Z0``; ``f`` in Hz.
    """
    T = np.asarray(T)
    A, B, D = T[:, 0, 0], T[:, 0, 1], T[:, 1, 1]
    n_cells = int(cell.N_tot / cell.N)
    root = np.sqrt((A + D) ** 2 - 4)
    Zb = -2 * B / (A - D - root)
    Zb.real = np.abs(Zb.real)
    return {
        "k": bloch_wavenumber(cell, f, T, Z0) / n_cells,
        "alpha": np.abs(np.log((A + D + root) / 2).real) / n_cells,
        "Zb": Zb,
        "reflection_coeff": (Zb - Z0) / (Zb + Z0),
    }


def linear_data(twpa: TWPA, f: np.ndarray, unit_multiplier: float) -> LinearData:
    """Linear response of ``twpa`` on the grid ``f`` (Hz), reported in ``f / unit_multiplier``."""
    segmented = is_segmented(twpa)
    segments: list[Any] = segments_of(twpa)
    if segmented:
        log.info("Detected %d TWPA segments.", len(segments))
    # Segments with identical unit cells share one supercell ABCD and its
    # Bloch parameters (N and the nonlinearity do not enter them).
    cache: dict[str, Any] = {}
    params: list[dict[str, Any]] = []
    abcd: Any = None
    for seg in segments:
        key = repr([c.model_dump() for c in seg.cells])
        if key not in cache:
            T = seg.single_abcd(f)
            cache[key] = (T, _bloch_params(seg, f, twpa.Z0_ref, T))
        T, p = cache[key]
        params.append(p)
        T_seg = T if seg.N == 1 else T**seg.N
        abcd = T_seg if abcd is None else abcd @ T_seg
    if segmented and twpa.N != 1:
        abcd = abcd**twpa.N
    cell = TwoPortCell(f, abcd=abcd, Z0=twpa.Z0_ref)
    freqs = f / unit_multiplier

    # One [start, end] row (first and last grid point) per contiguous band.
    gap = in_stopband(lossless_copy(segments[0]), f)
    edges = np.flatnonzero(np.diff(np.concatenate([[0], gap, [0]])))
    stopbands = freqs[edges.reshape(-1, 2) - [0, 1]]
    if not len(stopbands):
        log.warning("No stopband in the frequency grid.")

    tables = {
        key: np.stack([p[key] for p in params])
        for key in ("k", "alpha", "reflection_coeff", "Zb")
    }
    seg_N_tot = np.array([seg.N_tot for seg in segments], dtype=np.float64)
    k_full = np.average(tables["k"], axis=0, weights=seg_N_tot)
    freqs_mask = freqs < (stopbands[0, 0] / 10 if len(stopbands) else 0.0)
    k_star = k_full
    if freqs_mask.sum() > 2:
        k_pars = np.polyfit(freqs[freqs_mask], k_full[freqs_mask], 2)
        k_star = k_full - np.polyval(k_pars[1:], freqs)

    return LinearData(
        freqs=freqs,
        abcd=np.asarray(cell.abcd),
        S21=cell.s.S21,
        S11=cell.s.S11,
        stopbands=stopbands,
        k_star=k_star,
        seg_bounds=np.concatenate([[0.0], np.cumsum(seg_N_tot)]),
        seg_N_tot=seg_N_tot,
        seg_epsilon=np.array([seg.epsilon for seg in segments]),
        seg_xi=np.array([seg.xi for seg in segments]),
        **tables,
    )


def interpolator(lin: LinearData) -> ParameterInterpolator:
    """Spline of the per-segment ``k``, ``gamma`` (reflection), ``alpha`` and ``Zb``."""
    return ParameterInterpolator(
        lin.freqs,
        {"k": lin.k, "gamma": lin.reflection_coeff, "alpha": lin.alpha, "Zb": lin.Zb},
    )
