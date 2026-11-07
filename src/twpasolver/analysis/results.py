"""Results of the gain and phase-matching analyses."""

from dataclasses import asdict, dataclass, field
from typing import Any, Type, TypeVar

import numpy as np
from matplotlib.axes import Axes

from twpasolver.file_utils import read_file, save_to_file
from twpasolver.plotting import plot_gain, plot_mode_currents, plot_phase_matching

R = TypeVar("R", bound="_Saveable")


class _Saveable:
    """HDF5 save and load of a result dataclass."""

    def save(self, filename: str) -> None:
        """Save the result to an HDF5 file."""
        save_to_file(filename, asdict(self), writer="hdf5")  # type: ignore[call-overload]

    @classmethod
    def load(cls: Type[R], filename: str) -> R:
        """Load a result saved with :meth:`save`."""
        return cls(**read_file(filename, writer="hdf5"))


@dataclass
class GainResult(_Saveable):
    """
    Result of :meth:`TWPAnalysis.gain`; the same fields for every model.

    Frequencies are in ``unit``, positions in unit cells, currents in A.

    Attributes:
        model: Gain model, ``"simple"`` or ``"bloch"``.
        unit: Frequency unit.
        signal_freqs: Signal frequencies, shape (n_freq,).
        pump_freq: Pump frequency.
        x: Positions along the device, shape (n_x,).
        mode_labels: Mode names, in the order of the mode axis.
        signal_mode, pump_mode: Names of the signal and pump modes.
        I_fwd, I_bwd: Forward and backward mode currents inside the device,
            shape (n_freq, n_modes, n_x); ``I_bwd`` is zero for ``simple``.
        a_in, a_out: Currents incident from the external lines at x=0 and x=L,
            shape (n_freq, n_modes).
        b_in, b_out: Currents going out into the external lines at x=0 and x=L
            (``b_in`` is NaN for ``simple``, which does not model it).
        gain_db, gain_db_12, gain_db_11, gain_db_22: Signal S21, S12, S11 and S22
            in dB, shape (n_freq,); NaN where the reference input is zero. With
            inputs at both ends the outputs contain the response to both.
        converged: Points whose solution converged (``bloch``) or reached the end
            of the device (``simple``), shape (n_freq,).
        diagnostics: Solver tables (``bloch``: Bloch tables, edge-basis flags,
            ``residuals``). The forward-backward residual is the largest change of
            any mode between passes relative to that mode's largest forward plus
            backward envelope; nearly empty backward envelopes dominate it, so
            1e-3 to 1e-1 is common while the gains are stable.
        history: With ``BlochSolverConfig(save_all_passes=True)``, every pass:
            ``I_fwd``, ``I_bwd``, ``b_in``, ``b_out`` and the four gains with a
            leading pass axis of length ``passes``; NaN after the last pass a point
            ran (``diagnostics["passes_used"]``). Otherwise empty.
    """

    model: str
    unit: str
    signal_freqs: np.ndarray
    pump_freq: float
    x: np.ndarray
    mode_labels: list[str]
    signal_mode: str
    pump_mode: str
    I_fwd: np.ndarray
    I_bwd: np.ndarray
    a_in: np.ndarray
    a_out: np.ndarray
    b_in: np.ndarray
    b_out: np.ndarray
    gain_db: np.ndarray
    gain_db_12: np.ndarray
    gain_db_11: np.ndarray
    gain_db_22: np.ndarray
    converged: np.ndarray
    diagnostics: dict[str, Any] = field(default_factory=dict)
    history: dict[str, Any] = field(default_factory=dict)

    def mode(self, label: str) -> int:
        """Index of mode ``label`` on the mode axis."""
        return list(self.mode_labels).index(label)

    def bandwidth(self, gain_reduction: float = 3) -> dict[str, Any]:
        """
        Compound bandwidth of ``gain_db`` within ``gain_reduction`` dB of its maximum.

        Returns ``bandwidth_edges`` (start and end of every contiguous band),
        ``total_bw``, ``max_gain``, ``reduced_gain``, ``mean_gain`` (over the band)
        and ``bw_freqs``.
        """
        gain_db, freqs = self.gain_db, self.signal_freqs
        if np.all(np.isnan(gain_db)):
            raise ValueError("Gain is NaN at every frequency.")
        max_g = gain_db[np.nanargmax(gain_db)]
        ok_idx = np.where(gain_db >= max_g - gain_reduction)[0]
        ok_freqs = freqs[ok_idx]
        df = freqs[1] - freqs[0]
        edges = [ok_freqs[0]]
        for idx in np.where(np.diff(ok_freqs) >= 2 * df)[0]:
            edges += [ok_freqs[idx], ok_freqs[idx + 1]]
        edges.append(ok_freqs[-1])
        return {
            "pump_freq": self.pump_freq,
            "bandwidth_edges": edges,
            "total_bw": len(ok_freqs) * df,
            "max_gain": max_g,
            "reduced_gain": max_g - gain_reduction,
            "mean_gain": np.mean(gain_db[ok_idx]),
            "bw_freqs": ok_freqs,
        }

    def plot(self, **kwargs) -> Axes:
        """Plot ``gain_db`` versus signal frequency."""
        return plot_gain(
            self.signal_freqs, self.gain_db, freqs_unit=self.unit, **kwargs
        )

    def plot_currents(self, backward: bool = False, **kwargs) -> Axes:
        """Plot the mode currents along the device, averaged over frequency."""
        return plot_mode_currents(
            self.x,
            self.I_bwd if backward else self.I_fwd,
            list(self.mode_labels),
            pump_freq=self.pump_freq,
            freqs_unit=self.unit,
            **kwargs,
        )


@dataclass
class PhaseMatchingResult(_Saveable):
    """
    Result of :meth:`TWPAnalysis.phase_matching`.

    Attributes:
        unit: Frequency unit.
        process, mixing: Process (``"PA"``, ``"FCU"``, ``"FCD"``) and mixing
            (``"3WM"``, ``"4WM"``).
        signal_mode, pump_mode, idler_mode: Mode names.
        signal_freqs, pump_freqs: Grids, shapes (n_signal,) and (n_pump,).
        delta: Phase mismatch per unit cell (relative to the signal ``k`` if
            ``relative``), shape (n_signal, n_pump); NaN for pumps in a stopband.
        relative: Whether ``delta`` is divided by the signal wavenumber.
        optimal_pump_freq: Pump with the smallest mean ``|delta|``; NaN if none.
        f_triplets, k_triplets: Pump, signal and idler frequencies and
            (length-weighted) wavenumbers for every pump and signal, shape
            (n_pump * n_signal, 3).
    """

    unit: str
    process: str
    mixing: str
    signal_mode: str
    pump_mode: str
    idler_mode: str
    signal_freqs: np.ndarray
    pump_freqs: np.ndarray
    delta: np.ndarray
    relative: bool
    optimal_pump_freq: float
    f_triplets: np.ndarray
    k_triplets: np.ndarray

    def plot(self, **kwargs) -> Axes:
        """Plot ``delta`` over pump and signal frequency."""
        return plot_phase_matching(
            self.pump_freqs,
            self.signal_freqs,
            self.delta,
            freqs_unit=self.unit,
            **kwargs,
        )
