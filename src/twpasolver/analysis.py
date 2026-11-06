"""Main class used to simulate the response of TWPAs."""

import enum
from abc import ABC, abstractmethod
from functools import partial, wraps
from time import strftime
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
from matplotlib.axes import Axes
from pydantic import ConfigDict, Field, PrivateAttr, field_validator

from twpasolver.basemodel import BaseModel
from twpasolver.bonus_types import FloatArray
from twpasolver.cmes.base import no_reflections_cmes_solve
from twpasolver.cmes.bloch import bloch_wavenumber, in_stopband, lossless_copy
from twpasolver.cmes.bloch_fb import bloch_fb_segmented_cmes_solve, segmented_phase
from twpasolver.cmes.prep import (
    fb_stopbands,
    prepare_bloch_segmented,
    prepare_relations_coefficients,
)
from twpasolver.cmes.solver_config import BlochSolverConfig, RKSolverConfig
from twpasolver.file_utils import read_file, save_to_file
from twpasolver.frequency import Frequencies
from twpasolver.logger import log
from twpasolver.mathutils import to_dB
from twpasolver.models import TWPA
from twpasolver.modes_rwa import ModeArray, ModeArrayFactory, ParameterInterpolator
from twpasolver.plotting import (
    plot_gain,
    plot_mode_currents,
    plot_phase_matching,
    plot_response,
)
from twpasolver.twoport import TwoPortCell, TwoPortModel


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
    """
    ``20 log10 |out / ref|`` per frequency point, NaN where the reference input is zero.

    A port-resolved gain is defined only when its reference port is driven.
    """
    out = np.asarray(out)
    ref = np.broadcast_to(np.asarray(ref), out.shape)
    res = np.full(out.shape, np.nan)
    ok = np.abs(ref) > 0
    res[ok] = to_dB(np.abs(out[ok] / ref[ok]))
    return res


def _mode_info(mode_array, mode_labels, signal_mode, pump_mode, idler_mode) -> dict:
    """``mode_info`` entry of a gain result."""
    mode_map = {label: idx for idx, label in enumerate(mode_labels)}
    return {
        "mode_array": mode_array,
        "mode_map": mode_map,
        "reverse_mode_map": {idx: label for label, idx in mode_map.items()},
        "signal_mode": signal_mode,
        "pump_mode": pump_mode,
        "idler_mode": idler_mode,
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


def analysis_function(func):
    """Decorate analysis methods: refresh base data, run, store the result."""

    @wraps(func)
    def wrapper(self, *args, save=True, save_name: Optional[str] = None, **kwargs):
        function_name = func.__name__
        log.info("Running %s", function_name)
        self.update_base_data()
        result = func(self, *args, **kwargs)
        if save:
            if save_name is None:
                save_name = function_name
            if save_name in self.data.keys():
                log.info(
                    "Data for '%s' output already present in analysis data, will be overwritten.",
                    save_name,
                )
            self.data[save_name] = result
        return result

    return wrapper


class Analyzer(BaseModel, ABC):
    """Base class for structured analysis."""

    data_file: str = Field(default_factory=partial(strftime, "data_%m_%d_%H_%M_%S"))
    data: dict[str, Any] = Field(default_factory=dict, exclude=True)

    @abstractmethod
    def update_base_data(self) -> None:
        """Check and update base data of the class if necessary."""

    def save_data(self, writer: str = "hdf5") -> None:
        """Save data to file."""
        save_to_file(self.data_file, self.data, writer=writer)

    def load_data(self, filename: str, writer="hdf5") -> None:
        """Load data from file."""
        self.data = read_file(filename, writer=writer)

    @analysis_function
    def parameter_sweep(
        self, function: str, target: str, values: list, *args, **kwargs
    ) -> dict[str, list[Any]]:
        """Run an analysis function multiple times for different values of a parameter."""
        results: dict[str, list[Any]] = {}
        kwargs.update({"save": False})
        fn = getattr(self, function)
        target_savename = target
        for i, value in enumerate(values):
            kwargs.update({target: value})
            fn_res = fn(*args, **kwargs)
            if i == 0:
                if target in fn_res.keys():
                    target_savename = target + "_sweeped"
                    log.warning(
                        "Results dictionary contains key with target variable name, "
                        "sweep values saved as '%s'.",
                        target_savename,
                    )
                results.update({key: [item] for key, item in fn_res.items()})
                results[target_savename] = []
            else:
                for key, item in fn_res.items():
                    results[key].append(item)
            results[target_savename].append(value)
        return results


class TWPAnalysis(Analyzer, Frequencies):
    """
    TWPA Analysis Engine with Extended Coupled Mode Equations.

    Main interface for simulating TWPA nonlinear behavior. Solves coupled mode
    equations (CMEs) with support for arbitrary numbers of modes and automatic
    mode relationship management.
    """

    model_config = ConfigDict(validate_assignment=True, extra="forbid")
    twpa: TWPA
    port: Any = Field(
        default=None,
        description=(
            "Connector model (``TwoPortModel``) at x=0 (input boundary). When "
            "``port_out`` is None, this model is also used at x=L."
        ),
    )
    port_out: Any = Field(
        default=None,
        description=(
            "Connector model at x=L (output boundary).  Same format as ``port``.\n"
            "When None the input connector model ``port`` is used at both boundaries"
            "symmetrically, which is correct for identical connectors on both sides.\n"
            "NOTE: port 2 of this model always faces the twpa."
        ),
    )
    port_Z_ext: float = Field(
        default=50.0,
        description=(
            "External termination impedance used when computing Z_in from a "
            "TwoPortModel port.  Typically the characteristic impedance of the "
            "fridge coaxial lines (default 50 Ω)."
        ),
    )

    @field_validator("port", "port_out", mode="before")
    @classmethod
    def _accept_port_as_is(cls, v):
        """Pass TwoPortModel instances through without Pydantic coercion."""
        if v is None or isinstance(v, TwoPortModel):
            return v
        raise ValueError(f"port must be a TwoPortModel instance, got {type(v)}")

    _previous_state: dict[str, Any] = PrivateAttr()
    _mode_arrays: Dict[str, ModeArray] = PrivateAttr(default_factory=dict)

    @field_validator("twpa", mode="before", check_fields=True)
    @classmethod
    def load_model_from_file(cls, twpa: str | TWPA) -> TWPA:
        """Try loading twpa model from file if filename string is provided."""
        if isinstance(twpa, str):
            try:
                return TWPA.from_file(twpa)
            except Exception as exc:
                raise ValueError(
                    "Input string must be valid path to model file."
                ) from exc
        return twpa

    def _is_segmented(self) -> bool:
        """Check whether the twpa is composed entirely of TWPA-instance segments."""
        return bool(self.twpa.cells) and all(
            isinstance(cell, TWPA) for cell in self.twpa.cells
        )

    def _bloch_segments(self) -> tuple[list, np.ndarray]:
        """
        Segments and segment boundaries (unit cells) seen by the Bloch FB model.

        A uniform device is a single segment spanning the whole line.
        """
        segments = list(self.twpa.cells) if self._is_segmented() else [self.twpa]
        return segments, np.asarray(self.data["segments"]["boundaries"], dtype=float)

    def _bloch_params_for_cell(self, cell: TWPA, T: Any = None) -> dict[str, Any]:
        """
        Bloch parameters of a uniform segment, per unit cell.

        ``k`` (winding rule, :func:`bloch_wavenumber`), ``alpha``, the Bloch
        impedance ``Zb`` and its ``reflection_coeff`` against ``self.twpa.Z0_ref``.
        ``T`` is the supercell ABCD of ``cell`` on ``self.f`` when already known.
        """
        if T is None:
            T = cell.single_abcd(self.f)
        T = np.asarray(T)
        A, B, D = T[:, 0, 0], T[:, 0, 1], T[:, 1, 1]
        n_cells = int(cell.N_tot / cell.N)
        root = np.sqrt((A + D) ** 2 - 4)
        Zb = -2 * B / (A - D - root)
        Zb.real = np.abs(Zb.real)
        Z0 = self.twpa.Z0_ref
        return {
            "k": bloch_wavenumber(cell, self.f, T, Z0) / n_cells,
            "alpha": np.abs(np.log((A + D + root) / 2).real) / n_cells,
            "Zb": Zb,
            "reflection_coeff": (Zb - Z0) / (Zb + Z0),
        }

    def _build_port_tables(
        self,
        mode_params: dict,
        mode_labels: list[str],
        n_freq: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Per-boundary port tables for the forward-backward solver.

        Port convention: port 1 → external line (port_Z_ext), port 2 → TWPA.
        The input connector (x=0) is ``self.port``; the output connector (x=L)
        is ``self.port_out`` when set, otherwise ``self.port``.

        Returns
        -------
        Zout_in_table : (n_freq, n_modes) complex
            Zout = (D·Ze + B)/(C·Ze + A) at x=0.
        Zout_out_table : (n_freq, n_modes) complex
            Zout = (D·Ze + B)/(C·Ze + A) at x=L.
        abcd_in_table : (n_freq, n_modes, 4) complex
            [A,B,C,D] of input connector at x=0.
        abcd_out_table : (n_freq, n_modes, 4) complex
            [A,B,C,D] of output connector at x=L.

        Without ports all tables encode identity connectors, so Zout = Ze =
        ``port_Z_ext`` (not ``twpa.Z0_ref``, which would add a spurious boundary
        mismatch when the two differ).
        """
        n_modes = len(mode_labels)
        Ze = self.port_Z_ext

        Zout_default = np.full((n_freq, n_modes), complex(Ze), dtype=np.complex128)
        abcd_identity = np.zeros((n_freq, n_modes, 4), dtype=np.complex128)
        abcd_identity[:, :, 0] = 1.0  # A
        abcd_identity[:, :, 3] = 1.0  # D

        port_in = self.port
        port_out = self.port_out if self.port_out is not None else self.port

        if port_in is None and port_out is None:
            return (
                Zout_default,
                Zout_default.copy(),
                abcd_identity,
                abcd_identity.copy(),
            )

        Zout_in = Zout_default.copy()
        Zout_out = Zout_default.copy()
        abcd_in = abcd_identity.copy()
        abcd_out = abcd_identity.copy()

        for m_idx, m in enumerate(mode_labels):
            freqs_m = np.abs(mode_params[m]["freqs"]) * self.unit_multiplier
            for table_Zout, table_abcd, port_model in (
                (Zout_in, abcd_in, port_in),
                (Zout_out, abcd_out, port_out),
            ):
                if port_model is None:
                    continue
                abcd = np.asarray(port_model.get_abcd(freqs_m))
                A, B = abcd[:, 0, 0], abcd[:, 0, 1]
                C, D = abcd[:, 1, 0], abcd[:, 1, 1]
                table_Zout[:, m_idx] = (D * Ze + B) / (C * Ze + A)
                table_abcd[:, m_idx, 0] = A
                table_abcd[:, m_idx, 1] = B
                table_abcd[:, m_idx, 2] = C
                table_abcd[:, m_idx, 3] = D

        return Zout_in, Zout_out, abcd_in, abcd_out

    def update_base_data(self) -> None:
        """Update data extracted from the twpa model."""
        current_state = {
            "twpa": self.twpa.model_dump(),
            "f_list": self.f_list,
            "f_arange": self.f_arange,
            "unit": self.unit,
        }
        if (
            not hasattr(self, "_previous_state")
            or current_state != self._previous_state
        ):
            log.info("Computing base parameters.")
            segmented = self._is_segmented()
            # A uniform device is one segment.
            segments: list[Any] = list(self.twpa.cells) if segmented else [self.twpa]
            # Segments with identical unit cells share one supercell ABCD and its
            # Bloch parameters (N and the nonlinearity do not enter them).
            cache: dict[str, Any] = {}
            params: list[dict[str, Any]] = []
            abcd: Any = None
            for seg in segments:
                key = repr([c.model_dump() for c in seg.cells])
                if key not in cache:
                    T = seg.single_abcd(self.f)
                    cache[key] = (T, self._bloch_params_for_cell(seg, T))
                T, p = cache[key]
                params.append(p)
                T_seg = T if seg.N == 1 else T**seg.N
                abcd = T_seg if abcd is None else abcd @ T_seg
            if segmented and self.twpa.N != 1:
                abcd = abcd**self.twpa.N
            cell = TwoPortCell(self.f, abcd=abcd, Z0=self.twpa.Z0_ref)
            freqs = self.f / self.unit_multiplier

            self.data["freqs"] = freqs
            self.data["abcd"] = np.asarray(cell.abcd)
            self.data["S21"] = cell.s.S21
            self.data["S11"] = cell.s.S11
            self.data["S21_db"] = 20 * np.log10(np.abs(self.data["S21"]))

            # Stopbands of the (first segment's) lossless supercell, one
            # [start, end] row (first and last grid point) per contiguous band.
            gap = in_stopband(lossless_copy(segments[0]), self.f)
            edges = np.flatnonzero(np.diff(np.concatenate([[0], gap, [0]])))
            stopbands = freqs[edges.reshape(-1, 2) - [0, 1]]
            if not len(stopbands):
                log.warning("No stopband in the frequency grid.")
            self.data["stopbands"] = stopbands

            # Bloch parameters: (n_segments, n_freq) for segmented devices,
            # (n_freq,) for uniform ones.
            for key in ("k", "alpha", "reflection_coeff", "Zb"):
                stacked = np.stack([p[key] for p in params])
                self.data[key] = stacked if segmented else stacked[0]
            seg_N_tot = np.array([seg.N_tot for seg in segments], dtype=np.float64)
            if segmented:
                log.info("Detected %d TWPA segments.", len(segments))
            bounds = np.concatenate([[0.0], np.cumsum(seg_N_tot)])
            self.data["segments"] = {
                "boundaries": bounds,
                "midpoints": 0.5 * (bounds[:-1] + bounds[1:]),
                "N_tot": seg_N_tot,
                "epsilon": np.array([seg.epsilon for seg in segments]),
                "xi": np.array([seg.xi for seg in segments]),
            }

            # k_star from the length-weighted mean wavenumber per unit cell.
            k_full = np.average(
                np.atleast_2d(self.data["k"]), axis=0, weights=seg_N_tot
            )
            freqs_mask = freqs < (stopbands[0, 0] / 10 if len(stopbands) else 0.0)
            if freqs_mask.sum() > 2:
                k_pars = np.polyfit(freqs[freqs_mask], k_full[freqs_mask], 2)
                self.data["k_star"] = k_full - np.polyval(k_pars[1:], freqs)
            else:
                self.data["k_star"] = k_full

            self._previous_state = current_state

            self._update_all_mode_arrays()
            if "basic_3wm" not in self._mode_arrays:
                self._initialize_standard_mode_arrays()

    def _base_interpolator(self) -> ParameterInterpolator:
        """Interpolator of the current base data, shared by all mode arrays."""
        return ParameterInterpolator(
            self.data["freqs"],
            {
                "k": self.data["k"],
                "gamma": self.data["reflection_coeff"],
                "alpha": self.data["alpha"],
                "Zb": self.data["Zb"],
            },
        )

    def _initialize_standard_mode_arrays(self) -> None:
        """Initialize standard mode arrays using the computed base data."""
        interpolator = self._base_interpolator()
        self._mode_arrays["basic_3wm"] = ModeArrayFactory.create_basic(
            self.data, interpolator=interpolator
        )
        self._mode_arrays["basic_4wm"] = ModeArrayFactory.create_basic(
            self.data, three_wave=False, interpolator=interpolator
        )

    def _update_all_mode_arrays(self) -> None:
        """Update all registered mode arrays with current base data."""
        if not self._mode_arrays:
            return
        interpolator = self._base_interpolator()
        for mode_array in self._mode_arrays.values():
            mode_array.update_base_data(interpolator)

    def get_mode_array(self, config: str = "basic_3wm") -> ModeArray:
        """Get a mode array by configuration name."""
        if config not in self._mode_arrays:
            raise ValueError(f"Mode array configuration '{config}' not found")
        return self._mode_arrays[config]

    def add_mode_array(self, name: str, mode_array: ModeArray) -> None:
        """Add a custom mode array to the analyzer."""
        self._mode_arrays[name] = mode_array

    # Shared CME helpers

    def _setup_initial_conditions(
        self,
        mode_array: ModeArray,
        signal_mode: str,
        pump_mode: str,
        Is0: float,
        Ip0: float,
        initial_amplitudes: Optional[Union[List[float], np.ndarray]] = None,
        default_fill: float = 0,
    ) -> np.ndarray:
        """
        Complex initial amplitudes, shape (n_modes,).

        *initial_amplitudes* if given, otherwise *default_fill* with the pump
        set to ``Ip0`` and the signal to ``Is0``.
        """
        if initial_amplitudes is not None:
            return np.array(initial_amplitudes, dtype=np.complex128)
        n_modes = len(mode_array.modes)
        mode_labels = list(mode_array.modes.keys())
        y0 = np.full(n_modes, default_fill, dtype=np.complex128)
        y0[mode_labels.index(pump_mode)] = Ip0
        y0[mode_labels.index(signal_mode)] = Is0
        return y0

    @staticmethod
    def _broadcast_ic(y0: np.ndarray, n_freq: int) -> np.ndarray:
        """Broadcast a 1-D IC vector to a contiguous (n_freq, n_modes) array."""
        if y0.ndim == 1:
            return np.repeat(y0, n_freq).reshape((-1, n_freq)).T.copy()
        return np.ascontiguousarray(y0)

    @staticmethod
    def _apply_phase_carrier(
        I_array: np.ndarray,
        gammas: np.ndarray,
        x: np.ndarray,
        segment_midpoints: np.ndarray,
    ) -> np.ndarray:
        """
        Multiply slow-envelope amplitudes by the propagation phase exp(i·Φ_m(x)).

        κ_m(x) is piecewise-linear on ``segment_midpoints`` (constant for one
        segment); the cumulative trapezoid rule is exact for that interpolant,
        consistent with the Numba kernel.

        Parameters
        ----------
        I_array          : (n_freq, n_modes, n_x)
        gammas           : (n_freq, n_modes, n_seg)
        x                : (n_x,)
        segment_midpoints: (n_seg,)
        """
        kappa_at_x = np.array(
            [
                [
                    np.interp(x, segment_midpoints, gammas[i, j].imag)
                    for j in range(gammas.shape[1])
                ]
                for i in range(gammas.shape[0])
            ]
        )
        dx = np.diff(x)
        dphi = 0.5 * (kappa_at_x[..., :-1] + kappa_at_x[..., 1:]) * dx
        phi = np.concatenate(
            [np.zeros((*gammas.shape[:2], 1)), np.cumsum(dphi, axis=-1)], axis=-1
        )
        return I_array * np.exp(1j * phi)

    def _prepare_cme_inputs(
        self,
        mode_array: ModeArray,
        signal_freqs: FloatArray,
        pump: float,
        signal_mode: str,
        pump_mode: str,
    ) -> dict[str, Any]:
        """
        Inputs shared by the CME solvers; a uniform device is one segment.

        gammas/reflections/Zb_table (n_freq, n_modes, n_segments), coefficient
        tables (n_relations, n_segments).
        """
        mode_array.update_frequencies({pump_mode: pump})
        mode_params = mode_array.process_frequency_array(signal_mode, signal_freqs)
        mode_labels = list(mode_array.modes.keys())

        terms_3wm = mode_array.get_rwa_terms(power=2)
        terms_4wm = mode_array.get_rwa_terms(power=3)

        segments = self.data["segments"]
        n_seg = len(segments["N_tot"])

        def table(key):
            """(n_freq, n_modes, n_seg) from per-mode (n_seg, n_freq) parameters."""
            vals = [np.reshape(mode_params[m][key], (n_seg, -1)) for m in mode_labels]
            return np.stack(vals, axis=-2).transpose(2, 1, 0)

        # Relations are segment-independent; only epsilon and xi vary.
        relations_3wm, relations_4wm, c3_unit, c4_unit = prepare_relations_coefficients(
            terms_3wm, terms_4wm, 1.0, 1.0
        )
        coeffs_3wm = np.stack([c3_unit * eps for eps in segments["epsilon"]], axis=-1)
        coeffs_4wm = np.stack([c4_unit * xi for xi in segments["xi"]], axis=-1)
        gammas = -table("alpha") - 1j * table("k")
        reflections = table("gamma")
        Zb_table = table("Zb")
        segment_midpoints = segments["midpoints"]

        return {
            "mode_params": mode_params,
            "mode_labels": mode_labels,
            "n_freq": len(signal_freqs),
            "n_modes": len(mode_array.modes),
            "relations_3wm": relations_3wm,
            "relations_4wm": relations_4wm,
            "coeffs_3wm": coeffs_3wm,
            "coeffs_4wm": coeffs_4wm,
            "gammas": gammas,
            "reflections": reflections,
            "Zb_table": Zb_table,
            "segment_midpoints": segment_midpoints,
        }

    # Public analysis methods

    @analysis_function
    def phase_matching(
        self,
        process: Literal["PA", "FCU", "FCD"] = "PA",
        mixing: Literal["3WM", "4WM"] = "3WM",
        signal_mode: str = "s",
        pump_mode: str = "p",
        idler_mode: str = "i",
        mode_array_config: str = "basic_3wm",
        pump_Ip0: Optional[float] = None,
        *,
        signal_arange: Union[Tuple[float, float, float], List[float], np.ndarray],
        pump_arange: Union[Tuple[float, float, float], List[float], np.ndarray],
        relative=True,
    ) -> dict[str, Any]:
        """
        Phase mismatch of a 3WM or 4WM process over pump and signal frequencies.

        Parameters
        ----------
        pump_Ip0 : float, optional
            Pump current used for the Kerr term. None uses ``twpa.Ip0``; 0 gives
            the linear mismatch. Segmented devices give the length-weighted mean
            of the per-segment mismatch.
        signal_arange, pump_arange : tuple or array_like
            A tuple ``(start, stop, step)`` is passed to ``numpy.arange``; a list
            or array is used as the explicit frequencies.

        Returns
        -------
        dict
            Includes ``optimal_pump_freq``: the pump of ``pump_freqs`` with the
            smallest mean ``|delta|`` over ``signal_freqs`` for the chosen process
            and mixing (None if all NaN); restrict ``signal_arange`` to the band
            of interest. Pumps inside a stopband (``data["stopbands"]``) get a NaN
            ``delta`` column and are never chosen.
        """
        mode_array = self.get_mode_array(mode_array_config)
        pump_f = (
            np.arange(*pump_arange)
            if isinstance(pump_arange, tuple)
            else np.asarray(pump_arange, dtype=float)
        )
        signal_f = (
            np.arange(*signal_arange)
            if isinstance(signal_arange, tuple)
            else np.asarray(signal_arange, dtype=float)
        )

        Ip0 = self.twpa.Ip0 if pump_Ip0 is None else pump_Ip0
        # Segmented devices: the accumulated mismatch is sum_s N_s dk_s, so delta
        # is the length-weighted mean of the per-segment dk_s (own k, gamma, xi).
        seg = self.data["segments"]
        weights = seg["N_tot"] / seg["N_tot"].sum()
        chi = Ip0**2 * seg["xi"] / 8
        n_seg = len(weights)

        f_triplets = []
        k_triplets = []
        deltas = np.full((len(signal_f), len(pump_f)), np.nan)

        if process == "PA":
            idler_sign, signal_sign = -1, -1
        elif process == "FCD":
            idler_sign, signal_sign = 1, -1
        else:
            idler_sign, signal_sign = -1, 1

        pump_coeff = 2 if mixing == "4WM" else 1

        for i, p_freq in enumerate(pump_f):
            mode_array.update_frequencies({pump_mode: p_freq})
            mode_params = mode_array.process_frequency_array(signal_mode, signal_f)

            pump = mode_array.get_mode(pump_mode)
            kp = np.asarray(pump.k).reshape(n_seg, 1)
            g2 = np.abs(np.asarray(pump.gamma)).reshape(n_seg, 1) ** 2
            ks = np.asarray(mode_params[signal_mode]["k"]).reshape(n_seg, -1)
            ki = np.asarray(mode_params[idler_mode]["k"]).reshape(n_seg, -1)

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

            for j, (s_freq, i_freq) in enumerate(
                zip(mode_params[signal_mode]["freqs"], mode_params[idler_mode]["freqs"])
            ):
                f_triplets.append([p_freq, s_freq, i_freq])
                k_triplets.append([(weights @ kp).item(), ks_mean[j], ki_mean[j]])

        # A pump inside a stopband does not propagate: no phase matching there.
        stopbands = self.data["stopbands"]
        in_gap = (
            (pump_f[:, None] >= stopbands[:, 0]) & (pump_f[:, None] <= stopbands[:, 1])
        ).any(axis=1)
        deltas[:, in_gap] = np.nan

        valid = ~np.isnan(deltas).all(axis=0)
        mean_delta = np.full(len(pump_f), np.inf)
        mean_delta[valid] = np.nanmean(np.abs(deltas[:, valid]), axis=0)
        return {
            "delta": deltas,
            "triplets": {"f": f_triplets, "k": k_triplets},
            "pump_freqs": pump_f,
            "optimal_pump_freq": (
                pump_f[np.argmin(mean_delta)] if valid.any() else None
            ),
            "signal_freqs": signal_f,
            "mode_info": {
                "signal": signal_mode,
                "pump": pump_mode,
                "idler": idler_mode,
                "mode_array": mode_array_config,
            },
        }

    @analysis_function
    def gain(
        self,
        signal_freqs: FloatArray,
        pump: float,
        Is0: float = 1e-7,
        Ip0: Optional[float] = None,
        model: Union[str, GainModel] = GainModel.SIMPLE,
        mode_array_config: str = "basic_3wm",
        signal_mode: str = "s",
        pump_mode: str = "p",
        idler_mode: str = "i",
        initial_amplitudes_fwd: Optional[Union[List[float], np.ndarray]] = None,
        initial_amplitudes_bwd: Optional[Union[List[float], np.ndarray]] = None,
        thin: int = 100,
        bloch_config: Optional[BlochSolverConfig] = None,
        rk_config: Optional[RKSolverConfig] = None,
    ) -> dict[str, Any]:
        """
        Compute the TWPA gain using coupled mode equation models.

        Parameters
        ----------
        signal_freqs : array_like
            Signal frequencies, in the analysis unit.
        pump : float
            Pump frequency. There is no automatic choice; for an estimate use
            ``phase_matching(...)["optimal_pump_freq"]``.
        Is0 : float, default 1e-7
            Initial signal current amplitude (A).
        Ip0 : float, optional
            Initial pump current amplitude (A); defaults to ``twpa.Ip0``.
        model : GainModel or str, default 'simple'
            CME model (see :class:`GainModel`):
            * 'simple': traditional CMEs of a homogeneous line, forward waves
              only; fast, approximate couplings, no reflections or connectors.
            * 'bloch': forward-backward model in the Bloch-mode basis with exact
              couplings and boundary conditions; segmented devices are solved per
              segment with exact junction conditions.
        mode_array_config : str, default 'basic_3wm'
            Mode array name ('basic_3wm', 'basic_4wm' or one added with
            :meth:`add_mode_array`).
        initial_amplitudes_fwd, initial_amplitudes_bwd : array_like, optional
            Amplitudes of the waves injected at x=0 (traveling along +x) and at
            x=L (traveling along -x); override Is0 and Ip0. Backward inputs
            require the 'bloch' model; 'simple' raises a ValueError if any is
            nonzero.
        thin : int, default 100
            Spacing in unit cells of the position grid (at least two points).
        bloch_config : BlochSolverConfig, optional
            Settings of the 'bloch' solver (default ``BlochSolverConfig()``);
            ignored by 'simple'.
        rk_config : RKSolverConfig, optional
            Settings of the RK45 integrator of both models (default
            ``RKSolverConfig()``).

        Returns
        -------
        dict
            Gain analysis results. The 'bloch' model also returns the
            port-resolved signal gains ``gain_db`` (S21) and ``gain_db_11``
            (S11), referenced to the forward input, and ``gain_db_12`` (S12)
            and ``gain_db_22`` (S22), referenced to the backward input set with
            ``initial_amplitudes_bwd``. A gain whose reference input is zero is
            NaN. With inputs at both ends the outputs contain the response to
            both, so inject from one side at a time for S-parameters.
        """
        try:
            model = GainModel(model)
        except ValueError:
            raise ValueError(
                f"Invalid model: {model}. Must be one of {[m.value for m in GainModel]}"
            ) from None

        if (
            model == GainModel.SIMPLE
            and initial_amplitudes_bwd is not None
            and np.any(np.asarray(initial_amplitudes_bwd) != 0)
        ):
            raise ValueError(
                "Model 'simple' propagates only +x traveling waves and cannot use "
                "backward inputs (initial_amplitudes_bwd). Use model='bloch' for "
                "counter-propagating waves."
            )

        if Ip0 is None:
            Ip0 = self.twpa.Ip0

        signal_freqs = np.asarray(signal_freqs)
        N_tot = self.twpa.N_tot
        x = np.linspace(
            0, N_tot, max(2, int(N_tot / thin)), endpoint=True, dtype=np.float64
        )

        if model == GainModel.BLOCH:
            return self._compute_bloch_gain(
                signal_freqs,
                pump,
                Is0,
                Ip0,
                mode_array_config,
                signal_mode,
                pump_mode,
                idler_mode,
                x,
                initial_amplitudes_fwd,
                initial_amplitudes_bwd,
                bloch_config,
                rk_config,
            )
        return self._compute_simple_gain(
            signal_freqs,
            pump,
            mode_array_config,
            signal_mode,
            pump_mode,
            idler_mode,
            x,
            initial_amplitudes_fwd,
            Is0,
            Ip0,
            rk_config,
        )

    # Private CME solvers

    def _compute_simple_gain(
        self,
        signal_freqs: FloatArray,
        pump: float,
        mode_array_config: str,
        signal_mode: str,
        pump_mode: str,
        idler_mode: str,
        x: FloatArray,
        initial_amplitudes_fwd: Optional[Union[List[float], np.ndarray]] = None,
        Is0: float = 1e-6,
        Ip0: float = 0.0,
        rk_config: Optional[RKSolverConfig] = None,
    ) -> dict[str, Any]:
        """Gain with the traditional forward-only CMEs (``model="simple"``)."""
        mode_array = self.get_mode_array(mode_array_config)

        y0 = self._setup_initial_conditions(
            mode_array,
            signal_mode,
            pump_mode,
            Is0,
            Ip0,
            initial_amplitudes_fwd,
        )

        cme_inputs = self._prepare_cme_inputs(
            mode_array, signal_freqs, pump, signal_mode, pump_mode
        )
        mode_labels = cme_inputs["mode_labels"]
        n_freq = cme_inputs["n_freq"]
        gammas = cme_inputs["gammas"]

        I_tuples_array = no_reflections_cmes_solve(
            x,
            self._broadcast_ic(y0, n_freq),
            gammas,
            cme_inputs["relations_3wm"],
            cme_inputs["relations_4wm"],
            cme_inputs["coeffs_3wm"],
            cme_inputs["coeffs_4wm"],
            segment_midpoints=cme_inputs["segment_midpoints"],
            rk_config=rk_config,
        )

        I_tuples_array = self._apply_phase_carrier(
            I_tuples_array, gammas, x, cme_inputs["segment_midpoints"]
        )

        signal_idx = mode_labels.index(signal_mode)
        initial_signal = y0[signal_idx] if y0.ndim == 1 else y0[:, signal_idx]
        gain_db = 10 * np.log10(
            np.abs(I_tuples_array[:, signal_idx, -1] / initial_signal) ** 2
        )

        return {
            "model": GainModel.SIMPLE.value,
            "pump_freq": pump,
            "signal_freqs": signal_freqs,
            "x": x,
            "I_tuples": I_tuples_array,
            "gain_db": gain_db,
            "mode_info": _mode_info(
                mode_array_config, mode_labels, signal_mode, pump_mode, idler_mode
            ),
        }

    def _compute_bloch_gain(
        self,
        signal_freqs: FloatArray,
        pump: float,
        Is0: float,
        Ip0: float,
        mode_array_config: str,
        signal_mode: str,
        pump_mode: str,
        idler_mode: str,
        x: FloatArray,
        initial_amplitudes_fwd: Optional[Union[List[float], np.ndarray]] = None,
        initial_amplitudes_bwd: Optional[Union[List[float], np.ndarray]] = None,
        bloch_config: Optional[BlochSolverConfig] = None,
        rk_config: Optional[RKSolverConfig] = None,
    ) -> dict[str, Any]:
        """
        Gain with the forward-backward CMEs in the Bloch basis (``model="bloch"``).

        Amplitudes are +x-referenced boundary currents of the forward and
        backward Bloch modes; the constant boundary coefficients use the exact
        Bloch impedances Zb+ (forward) and Zb- (backward). See
        :mod:`twpasolver.cmes.bloch_fb`.
        """
        segmented = self._is_segmented()
        bloch_config = bloch_config if bloch_config is not None else BlochSolverConfig()
        mode_array = self.get_mode_array(mode_array_config)

        y0_fwd = self._setup_initial_conditions(
            mode_array,
            signal_mode,
            pump_mode,
            Is0,
            Ip0,
            initial_amplitudes_fwd,
            default_fill=0.0,
        )
        y0_bwd = self._setup_initial_conditions(
            mode_array,
            signal_mode,
            pump_mode,
            Is0=0.0,
            Ip0=0.0,
            initial_amplitudes=initial_amplitudes_bwd,
            default_fill=0.0,
        )

        cme_inputs = self._prepare_cme_inputs(
            mode_array, signal_freqs, pump, signal_mode, pump_mode
        )
        mode_labels = cme_inputs["mode_labels"]
        n_freq = cme_inputs["n_freq"]
        mode_params = cme_inputs["mode_params"]
        segments, seg_bounds = self._bloch_segments()
        bloch = prepare_bloch_segmented(
            mode_array, cme_inputs, segments, seg_bounds, self.unit_multiplier
        )

        Zout_in, Zout_out, abcd_in, abcd_out = self._build_port_tables(
            mode_params, mode_labels, n_freq
        )

        y0_fwd_broadcast = self._broadcast_ic(y0_fwd, n_freq)
        y0_bwd_broadcast = self._broadcast_ic(y0_bwd, n_freq)
        injected = np.any(np.abs(y0_fwd_broadcast) > 0, axis=0) | np.any(
            np.abs(y0_bwd_broadcast) > 0, axis=0
        )
        sb = fb_stopbands(
            bloch, cme_inputs, cme_inputs["gammas"], injected, bloch_config, segments
        )
        gammas, edge = sb["gammas"], sb["edge"]
        diag: dict[str, Any] = {}
        I_tuples_raw = bloch_fb_segmented_cmes_solve(
            x,
            bloch["seg_bounds"],
            y0_fwd_broadcast,
            y0_bwd_broadcast,
            np.transpose(gammas, (0, 2, 1)),
            cme_inputs["relations_3wm"],
            cme_inputs["relations_4wm"],
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
            Ze=self.port_Z_ext,
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
        # Exact phase of the piecewise-constant wavenumber (k x for one segment).
        Phi_f, Phi_b = segmented_phase(gammas.imag, bloch["seg_bounds"], x)
        carrier_fwd = np.exp(1j * Phi_f)
        carrier_bwd = np.exp(1j * Phi_b)
        # Launched waves: forward mode of the first segment at x=0,
        # backward mode of the last segment at x=L.
        Zp_launch = bloch["Zb_plus"][:, 0]
        Zm_launch = bloch["Zb_minus"][:, -1]
        if not segmented:
            bloch = _drop_segment_axis(bloch, n_freq)

        num_saved_passes = bloch_config.passes if bloch_config.save_all_passes else 1
        I_tuples_all_passes = {}
        for p in range(num_saved_passes):
            I_tuples_all_passes[p] = {
                "forward": I_tuples_raw[p * n_freq : (p + 1) * n_freq] * carrier_fwd,
                "backward": I_tuples_raw[
                    num_saved_passes * n_freq
                    + p * n_freq : num_saved_passes * n_freq
                    + (p + 1) * n_freq
                ]
                * carrier_bwd,
            }

        I_tuples_fwd = I_tuples_all_passes[num_saved_passes - 1]["forward"]
        I_tuples_bwd = I_tuples_all_passes[num_saved_passes - 1]["backward"]

        signal_idx = mode_labels.index(signal_mode)

        # Prompt reflections of the injected waves: the wave launched at x=0
        # is the forward Bloch mode (Zb+), the one launched at x=L is the
        # backward mode (Zb-).
        Ze = self.port_Z_ext
        Zb_launch_in = Zp_launch[:, signal_idx]
        Zb_launch_out = Zm_launch[:, signal_idx]

        A_in, B_in, C_in, D_in = (abcd_in[:, signal_idx, k] for k in range(4))
        Zin_in = (A_in * Zb_launch_in + B_in) / (C_in * Zb_launch_in + D_in)
        refl_in = (Ze - Zin_in) / (Ze + Zin_in)

        A_out, B_out, C_out, D_out = (abcd_out[:, signal_idx, k] for k in range(4))
        Zin_out = (A_out * Zb_launch_out + B_out) / (C_out * Zb_launch_out + D_out)
        refl_out = (Ze - Zin_out) / (Ze + Zin_out)

        Is0_rep = y0_fwd_broadcast[:, signal_idx]
        Ib0_rep = y0_bwd_broadcast[:, signal_idx]

        def _gains(I_fwd_arr, I_bwd_arr):
            out_fwd = (
                I_fwd_arr[:, signal_idx, -1]
                + y0_bwd_broadcast[:, signal_idx] * refl_out
            )
            out_bwd = (
                I_bwd_arr[:, signal_idx, -1] + y0_fwd_broadcast[:, signal_idx] * refl_in
            )
            return {
                "gain_db_fwd": _ratio_db(out_fwd, Is0_rep),
                "gain_db_bwd": _ratio_db(out_bwd, Ib0_rep),
                "gain_db_11": _ratio_db(out_bwd, Is0_rep),
                "gain_db_22": _ratio_db(out_fwd, Ib0_rep),
            }

        final = _gains(I_tuples_fwd, I_tuples_bwd)
        gains_all_passes = [
            _gains(
                I_tuples_all_passes[p]["forward"], I_tuples_all_passes[p]["backward"]
            )
            for p in range(num_saved_passes)
        ]

        return {
            "model": GainModel.BLOCH.value,
            "pump_freq": pump,
            "signal_freqs": signal_freqs,
            "x": x,
            "I_tuples": I_tuples_fwd,
            "I_tuples_fwd": I_tuples_fwd,
            "I_tuples_bwd": I_tuples_bwd,
            "I_tuples_all_passes": I_tuples_all_passes,
            "gain_db": final["gain_db_fwd"],
            "gain_db_12": final["gain_db_bwd"],
            "gain_db_11": final["gain_db_11"],
            "gain_db_22": final["gain_db_22"],
            "gains_all_passes": gains_all_passes,
            "passes": bloch_config.passes,
            "bloch": bloch,
            "mode_info": _mode_info(
                mode_array_config, mode_labels, signal_mode, pump_mode, idler_mode
            ),
        }

    # Bandwidth

    @analysis_function
    def bandwidth(
        self,
        gain_reduction: float = 3,
        **gain_kwargs,
    ) -> dict[str, Any]:
        """
        Compound gain bandwidth of the stored (or newly computed) gain.

        Parameters
        ----------
        gain_reduction : float
            Threshold in dB below the maximum gain.
        **gain_kwargs : dict
            If given, ``gain()`` is run with them first.
        """
        if gain_kwargs:
            self.gain(**gain_kwargs)
        elif "gain" not in self.data.keys():
            raise ValueError(
                "No stored gain: run gain() first or pass its arguments to bandwidth()."
            )

        gain_data = self.data["gain"]
        gain_db = gain_data["gain_db"]
        signal_freqs = gain_data["signal_freqs"]
        if np.all(np.isnan(gain_db)):
            raise ValueError("Stored gain is NaN at every frequency.")

        max_g = gain_db[np.nanargmax(gain_db)]
        ok_idx = np.where(gain_db >= max_g - gain_reduction)[0]
        ok_freqs = signal_freqs[ok_idx]
        df = signal_freqs[1] - signal_freqs[0]

        extremes_indices = np.where(np.diff(ok_freqs) >= 2 * df)[0]
        all_roots = [ok_freqs[0]]
        for idx in extremes_indices:
            all_roots.extend([ok_freqs[idx], ok_freqs[idx + 1]])
        all_roots.append(ok_freqs[-1])

        return {
            "pump_freq": gain_data["pump_freq"],
            "bandwidth_edges": all_roots,
            "total_bw": len(ok_freqs) * df,
            "max_gain": max_g,
            "reduced_gain": max_g - gain_reduction,
            "mean_gain": np.mean(gain_db[ok_idx]),
            "bw_freqs": ok_freqs,
        }

    # Plotting helpers

    def plot_response(self, **kwargs) -> Axes:
        """Plot response of twpa."""
        if "k_star" not in self.data:
            self.update_base_data()
        return plot_response(
            self.data["freqs"],
            self.data["S21_db"],
            self.data["k_star"],
            freqs_unit=self.unit,
            **kwargs,
        )

    def plot_gain(self, **kwargs) -> Axes:
        """Plot gain of twpa."""
        if "gain" not in self.data:
            raise RuntimeError("Gain data not found, please run analysis function.")
        gain_data = self.data["gain"]
        return plot_gain(
            gain_data["signal_freqs"],
            gain_data["gain_db"],
            freqs_unit=self.unit,
            **kwargs,
        )

    def plot_mode_currents(self, **kwargs) -> Axes:
        """Plot mean currents of modes as a function of position along the TWPA line."""
        if "gain" not in self.data:
            raise RuntimeError("Gain data not found, please run analysis function.")
        return plot_mode_currents(self.data["gain"], **kwargs)

    def plot_phase_matching(self, **kwargs) -> Axes:
        """Plot phase matching profile of twpa."""
        if "phase_matching" not in self.data:
            raise RuntimeError(
                "Phase matching data not found, please run analysis function."
            )
        pm = self.data["phase_matching"]
        return plot_phase_matching(
            pm["pump_freqs"],
            pm["signal_freqs"],
            pm["delta"],
            freqs_unit=self.unit,
            **kwargs,
        )
