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
from twpasolver.cmes import solver_config
from twpasolver.cmes.base import no_reflections_cmes_solve
from twpasolver.cmes.bloch import (
    bloch_coefficients,
    bloch_mode_tables,
    bloch_no_reflections_cmes_solve,
    cell_phase_mask,
    edge_basis_tables,
    edge_guard_mode_weights,
    edge_proximity_mode_weights,
    edge_reference_frequency,
    lossless_copy,
    process_weights,
    reactive_angle_deg,
)
from twpasolver.cmes.bloch_fb import bloch_fb_cmes_solve
from twpasolver.cmes.bloch_fb_segmented import (
    bloch_fb_segmented_cmes_solve,
    segmented_phase,
)
from twpasolver.cmes.cmes_fb import general_cmes_solve_fb_cutoff
from twpasolver.cmes.solver_config import FBSolverConfig
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

    Args
    ----
    GENERAL_NO_REFLECTIONS : str
        Extended mode system without reflections.
        Fast and useful for initial assessment of the main nonlinear processes.

    GENERAL : str
        Forward-backward propagation model with iterative solving for reflections.
        Most complete model modeling impedance mismatched at the edges of the device.

    BLOCH_NO_REFLECTIONS : str
        Extended mode system without reflections, written in the Bloch-mode basis.
        Coupling coefficients include the Bloch-impedance normalization and the
        intracell current profiles of the modes (relevant close to stopbands).
        Currently available only for non-segmented devices.

    BLOCH : str
        Forward-backward model in the Bloch-mode basis. Same couplings as
        BLOCH_NO_REFLECTIONS, plus exact boundary conditions at the device ends
        (Bloch impedances of the forward and backward modes, no derivative
        correction). Segmented devices are solved piecewise, each segment on
        its own Bloch modes, joined by exact junction conditions.
    """

    GENERAL_NO_REFLECTIONS = "general_no_reflections"
    GENERAL = "general"
    BLOCH_NO_REFLECTIONS = "bloch_no_reflections"
    BLOCH = "bloch"


def _ratio_db(out: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """
    ``20 log10 |out / ref|`` per frequency point, NaN where the reference input is zero.

    Used for the port-resolved gains of the forward-backward models: S12 and
    S22 are defined only when a signal is injected from the output side (and
    S21, S11 only when it is injected from the input side), so an absent
    reference input gives NaN instead of a division by zero.
    """
    out = np.asarray(out)
    ref = np.broadcast_to(np.asarray(ref), out.shape)
    res = np.full(out.shape, np.nan)
    ok = np.abs(ref) > 0
    res[ok] = to_dB(np.abs(out[ok] / ref[ok]))
    return res


def prepare_relations_coefficients(terms_3wm, terms_4wm, epsilon, xi):
    """
    Prepare relation terms and coefficients for 3-wave and 4-wave mixing.

    Parameters
    ----------
    terms_3wm : list
        List of 3-wave mixing terms, each containing mode indices and coefficient.
    terms_4wm : list
        List of 4-wave mixing terms, each containing mode indices and coefficient.
    epsilon : float
        Scaling factor for 3-wave mixing coefficients.
    xi : float
        Scaling factor for 4-wave mixing coefficients.

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


def analysis_function(func):
    """Decorate analysis methods providing automatic result caching and management."""

    @wraps(func)
    def wrapper(self, *args, save=True, save_name: Optional[str] = None, **kwargs):
        function_name = func.__name__
        log.info("Running %s", function_name)
        result = func(self, *args, **kwargs)
        if save:
            if function_name in self.data.keys():
                log.info(
                    "Data for '%s' output already present in analysis data, will be overwritten.",
                    function_name,
                )
            if save_name is None:
                save_name = function_name
                kwargs.update({"save_name": save_name})
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
        results: dict[str, list[Any]] = {target: []}
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
                        "Results dictionary contains key with target variable name, will be overwritten.",
                        target_savename,
                    )
                results.update({key: [item] for key, item in fn_res.items()})
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

    model_config = ConfigDict(validate_assignment=True)
    twpa: TWPA
    port: Any = Field(
        default=None,
        description=(
            "Connector model at x=0 (input boundary). Accepts either a "
            "``TwoPortModel`` instance or a precomputed list of per-frequency "
            "impedances. When ``port_out`` is None, this model is also used at x=L."
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
        """Pass TwoPortModel instances and lists through without Pydantic coercion."""
        if v is None or isinstance(v, (list, TwoPortModel)):
            return v
        raise ValueError(
            f"port must be a TwoPortModel instance or a list of impedances, got {type(v)}"
        )

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

    def _bloch_params_for_cell(
        self,
        cell: TWPA,
        Z0_override: complex | float | None = None,
        abcd_single: Any = None,
    ) -> dict[str, Any]:
        """Extract Bloch-mode propagation parameters for a single uniform TWPA segment.

        Parameters
        ----------
        cell : TWPA
            The cell whose ABCD matrix is used to compute Bloch parameters.
        Z0_override : complex | float, optional
            Reference impedance to use for the reflection coefficient and Bloch
            impedance computation.  When provided it overrides ``cell.Z0_ref``.
            This should be set to the parent TWPA's ``Z0_ref`` when computing
            per-segment parameters so that all segments share the same reference
            as the full device.
        abcd_single : ABCDArray, optional
            Precomputed ABCD matrix of one period of ``cell`` on ``self.f``
            (``cell.single_abcd(self.f)``), to avoid computing it twice.
        """
        Z0 = Z0_override if Z0_override is not None else cell.Z0_ref
        if abcd_single is None:
            abcd_single = cell.single_abcd(self.f)
        A, B, C, D = abcd_single.A, abcd_single.B, abcd_single.C, abcd_single.D
        ad = A + D
        freqs = self.f / self.unit_multiplier

        gamma_bloch = np.log((ad + np.sqrt(ad**2 - 4)) / 2)
        Zb = -2 * B / (A - D - np.sqrt((A + D) ** 2 - 4))
        gamma_bloch.real = np.abs(gamma_bloch.real) / int(cell.N_tot / cell.N)
        gamma_bloch.imag = (
            np.unwrap(gamma_bloch.imag * np.sign(Zb.real) * 2)
            / 2
            / int(cell.N_tot / cell.N)
        )
        Zb.real = np.abs(Zb.real)

        reflection_coeff = (Zb - Z0) / (Zb + Z0)
        k = gamma_bloch.imag
        alpha = gamma_bloch.real
        vg = np.nan_to_num(np.gradient(k, freqs, axis=0), nan=0)

        return {
            "Zb": Zb,
            "gamma_bloch": gamma_bloch,
            "reflection_coeff": reflection_coeff,
            "k": k,
            "alpha": alpha,
            "vg": vg,
            "N_tot": cell.N_tot,
        }

    def _build_port_tables(
        self,
        mode_params: dict,
        mode_labels: list[str],
        n_freq: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Build per-boundary port tables for the forward-backward solver.

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

        When both port fields are None all tables encode identity connectors
        (A=1,B=0,C=0,D=1, Zout=Ze=self.port_Z_ext), recovering bare behaviour.
        Note the identity default is ``port_Z_ext``, not ``self.twpa.Z0_ref`` --
        for an identity connector Zout=(D*Ze+B)/(C*Ze+A)=Ze exactly, so using
        anything else here would silently introduce a spurious boundary
        mismatch whenever twpa.Z0_ref != port_Z_ext.
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
                if isinstance(port_model, TwoPortModel):
                    abcd = np.asarray(port_model.get_abcd(freqs_m))
                    A, B = abcd[:, 0, 0], abcd[:, 0, 1]
                    C, D = abcd[:, 1, 0], abcd[:, 1, 1]
                    table_Zout[:, m_idx] = (D * Ze + B) / (C * Ze + A)
                    table_abcd[:, m_idx, 0] = A
                    table_abcd[:, m_idx, 1] = B
                    table_abcd[:, m_idx, 2] = C
                    table_abcd[:, m_idx, 3] = D
                else:
                    table_Zout[:, m_idx] = np.interp(
                        freqs_m, self.data["freqs"] * self.unit_multiplier, port_model
                    )

        return Zout_in, Zout_out, abcd_in, abcd_out

    def update_base_data(self) -> None:
        """Update data extracted from the twpa model."""
        current_state = self.twpa.model_dump()
        if (
            not hasattr(self, "_previous_state")
            or current_state != self._previous_state
        ):
            log.info("Computing base parameters.")
            segmented = self._is_segmented()
            if segmented:
                # One supercell ABCD per distinct segment, reused for the device
                # ABCD and for the segment's Bloch parameters.
                segments = [c for c in self.twpa.cells if isinstance(c, TWPA)]
                seg_abcd = self._segment_supercell_abcds(segments)
                device_abcd = None
                for seg, T in zip(segments, seg_abcd):
                    T_seg = T if seg.N == 1 else T**seg.N
                    device_abcd = T_seg if device_abcd is None else device_abcd @ T_seg
                abcd_single = device_abcd
            else:
                abcd_single = self.twpa.single_abcd(self.f)
            cell = TwoPortCell(
                self.f,
                abcd=abcd_single if self.twpa.N == 1 else abcd_single**self.twpa.N,
                Z0=self.twpa.Z0_ref,
            )
            freqs = self.f / self.unit_multiplier

            # S-parameters: always from the full device.
            self.data["freqs"] = freqs
            self.data["abcd"] = np.asarray(cell.abcd)
            self.data["S21"] = cell.s.S21
            self.data["S11"] = cell.s.S11
            s21_db = 20 * np.log10(np.abs(self.data["S21"]))
            self.data["S21_db"] = s21_db

            s21_db_diff = s21_db[1:] - s21_db[:-1]
            stopband_start_idx = np.argmin(s21_db_diff)
            self.data["stopband_freqs"] = [
                freqs[stopband_start_idx],
                freqs[np.argmax(s21_db_diff)],
            ]

            # Bloch parameters: 2-D (n_segments, n_freq) for segmented devices,
            # 1-D (n_freq,) for uniform ones. ParameterInterpolator handles both
            # shapes identically, so _update_all_mode_arrays needs no branching.
            if segmented:
                log.info(
                    "Detected %d TWPA segments, computing Bloch parameters per segment.",
                    len(segments),
                )
                seg_results = self._segment_bloch_params(segments, seg_abcd)
                seg_N_tot = np.array(
                    [r["N_tot"] for r in seg_results], dtype=np.float64
                )
                seg_boundaries = np.concatenate([[0.0], np.cumsum(seg_N_tot)])

                self.data["k"] = np.stack([r["k"] for r in seg_results])
                self.data["alpha"] = np.stack([r["alpha"] for r in seg_results])
                self.data["reflection_coeff"] = np.stack(
                    [r["reflection_coeff"] for r in seg_results]
                )
                self.data["Zb"] = np.stack([r["Zb"] for r in seg_results])
                self.data["segments"] = {
                    "boundaries": seg_boundaries,
                    "midpoints": 0.5 * (seg_boundaries[:-1] + seg_boundaries[1:]),
                    "N_tot": seg_N_tot,
                    "epsilon": np.array([seg.epsilon for seg in segments]),  # type: ignore
                    "xi": np.array([seg.xi for seg in segments]),  # type: ignore
                }
                # Device-averaged wavenumber (per unit cell) for k_star and the
                # pump estimate: the length-weighted mean of the segment
                # wavenumbers. Unwrapping the phase of the whole device as a
                # single "supercell" would lose 2*pi windings on any practical
                # frequency grid.
                k_full = (seg_N_tot @ self.data["k"]) / seg_N_tot.sum()
            else:
                whole = self._bloch_params_for_cell(self.twpa, abcd_single=abcd_single)
                self.data["k"] = whole["k"]
                self.data["alpha"] = whole["alpha"]
                self.data["reflection_coeff"] = whole["reflection_coeff"]
                self.data["Zb"] = whole["Zb"]
                self.data.pop("segments", None)
                k_full = whole["k"]

            freqs_mask = freqs < freqs[stopband_start_idx] / 10
            if len(freqs_mask) > 2:
                k_pars = np.polyfit(freqs[freqs_mask], k_full[freqs_mask], 2)
                self.data["k_star"] = k_full - np.polyval(k_pars[1:], freqs)
            else:
                self.data["k_star"] = k_full

            self.data["optimal_pump_freq"] = self._estimate_optimal_pump()
            self._previous_state = current_state

            self._update_all_mode_arrays()
            if "basic_3wm" not in self._mode_arrays:
                self._initialize_standard_mode_arrays()

    def _segment_supercell_abcds(self, segments: list) -> list:
        """
        Supercell ABCD matrix of every segment on ``self.f``.

        Segments with identical unit cells share one computation (the
        repetition count ``N`` and the nonlinearity do not enter the linear
        supercell).
        """
        cache: dict[str, Any] = {}
        out = []
        for seg in segments:
            key = repr([c.model_dump() for c in seg.cells])
            if key not in cache:
                cache[key] = seg.single_abcd(self.f)
            out.append(cache[key])
        self._n_distinct_segments = len(cache)
        return out

    def _segment_bloch_params(self, segments: list, seg_abcd: list) -> list:
        """Bloch parameters of every segment from its precomputed supercell ABCD."""
        cache: dict[int, dict[str, Any]] = {}
        out = []
        for seg, T in zip(segments, seg_abcd):
            key = id(T)  # identical cells share the same ABCD object
            if key not in cache:
                cache[key] = self._bloch_params_for_cell(
                    seg, Z0_override=self.twpa.Z0_ref, abcd_single=T
                )
            res = dict(cache[key])
            res["N_tot"] = seg.N_tot
            out.append(res)
        return out

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

    def _estimate_optimal_pump(self) -> float:
        """Estimate optimal pump frequency for 3WM."""
        freqs = self.data["freqs"]
        # k_star is always 1-D (whole-device, used for response/phase-matching plots).
        ks = self.data["k_star"]
        min_p_idx = np.where(freqs == self.data["stopband_freqs"][1])[0][0]
        pump_f = freqs[min_p_idx:-1]
        pump_k = ks[min_p_idx:-1]
        signal_f = pump_f / 2
        signal_k = np.interp(signal_f, freqs, ks)
        deltas = np.abs(
            pump_k - 2 * signal_k + 1 * self.twpa.chi * (pump_k - 4 * signal_k)
        )
        return pump_f[np.argmin(deltas)]

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
        Build a 1-D complex128 IC vector for CME solving.

        If *initial_amplitudes* is provided it is returned as-is (cast to
        complex128).  Otherwise a vector of *default_fill* is constructed with
        the pump amplitude set to ``Ip0`` and the signal to *Is0*.
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
        segment_midpoints: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Multiply slow-envelope amplitudes by the propagation phase exp(i·Φ_m(x)).

        Uniform device (gammas.ndim == 2, shape (n_freq, n_modes)):
            Φ_m(x) = κ_m · x  (exact for constant κ_m).

        Segmented device (gammas.ndim == 3, shape (n_freq, n_modes, n_seg)):
            κ_m(x) is piecewise-linear on *segment_midpoints*, so the cumulative
            trapezoid rule on the interpolated grid is exact for that interpolant,
            consistent with the phi_table computed inside the segmented Numba kernel.

        Parameters
        ----------
        I_array          : (n_freq, n_modes, n_x)
        gammas           : (n_freq, n_modes) or (n_freq, n_modes, n_seg)
        x                : (n_x,)
        segment_midpoints: (n_seg,), required when gammas.ndim == 3
        """
        if gammas.ndim == 3:
            if segment_midpoints is None:
                raise ValueError("segment_midpoints is required when gammas.ndim == 3.")
            # Interpolate κ_m onto the output grid: (n_freq, n_modes, n_x)
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
        else:
            return I_array * np.exp(np.einsum("ij,k->ijk", 1j * gammas.imag, x))

    def _prepare_cme_inputs(
        self,
        mode_array: ModeArray,
        signal_freqs: FloatArray,
        pump: float,
        signal_mode: str,
        pump_mode: str,
    ) -> dict[str, Any]:
        """
        Prepare all inputs shared by the CME solvers.

        For uniform devices, mode_params values have shape (n_freq,) and
        gammas/reflections are (n_freq, n_modes).  For segmented devices the
        2-D interpolator returns (n_segments, n_freq) per mode, so gammas and
        reflections become (n_freq, n_modes, n_segments) and coefficient tables
        are (n_relations, n_segments).
        """
        mode_array.update_frequencies({pump_mode: pump})
        mode_params = mode_array.process_frequency_array(signal_mode, signal_freqs)
        mode_labels = list(mode_array.modes.keys())

        terms_3wm = mode_array.get_rwa_terms(power=2)
        terms_4wm = mode_array.get_rwa_terms(power=3)

        complex_gammas = [
            -mode_params[m]["alpha"] - 1j * mode_params[m]["k"] for m in mode_labels
        ]
        gamma_vals = [mode_params[m]["gamma"] for m in mode_labels]
        Zb_vals = [mode_params[m]["Zb"] for m in mode_labels]

        segments = self.data.get("segments")
        if segments is not None:
            # Relations are segment-independent. Only the scalar
            # nonlinearity coefficients (epsilon, xi) vary.
            relations_3wm, relations_4wm, c3_unit, c4_unit = (
                prepare_relations_coefficients(terms_3wm, terms_4wm, 1.0, 1.0)
            )
            coeffs_3wm = np.stack(
                [c3_unit * eps for eps in segments["epsilon"]], axis=-1
            )
            coeffs_4wm = np.stack([c4_unit * xi for xi in segments["xi"]], axis=-1)
            # mode_params[m]["k"] is (n_segments, n_freq) from the 2-D interpolator;
            # final shapes: (n_freq, n_modes, n_segments)
            gammas = np.stack(complex_gammas, axis=-2).transpose(2, 1, 0)
            reflections = np.stack(gamma_vals, axis=-2).transpose(2, 1, 0)
            Zb_table = np.stack(Zb_vals, axis=-2).transpose(2, 1, 0)
            segment_midpoints = segments["midpoints"]
        else:
            relations_3wm, relations_4wm, coeffs_3wm, coeffs_4wm = (
                prepare_relations_coefficients(
                    terms_3wm, terms_4wm, self.twpa.epsilon, self.twpa.xi
                )
            )
            gammas = np.stack(complex_gammas, axis=-1)
            reflections = np.stack(gamma_vals, axis=-1)
            Zb_table = np.stack(Zb_vals, axis=-1)
            segment_midpoints = None

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

    def _bloch_mode_weights(
        self,
        bloch: dict[str, Any],
        mode_labels: list[str],
        signal_freqs: FloatArray,
        injected: np.ndarray,
        forward_only: bool,
        keep: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Per-mode stopband-edge weights of the Bloch models.

        Works on tables of shape (n_freq, n_modes) or (n_freq, n_seg, n_modes).

        * Edge guard (all Bloch models, ``solver_config.BLOCH_EDGE_GUARD``):
          weight from the Bloch splitting ``|sinh(gamma d)|``, which vanishes
          only at the band edges (exceptional points). Modes inside a stopband,
          away from its edges, keep weight 1.
        * Reactive-angle guard (forward-only model only,
          ``solver_config.BLOCH_EDGE_GUARD_DEG``): weight from ``|arg W|``,
          which switches off a whole stopband, since a single evanescent Bloch
          mode cannot be described without its partner.

        Modes with an input amplitude (``injected``) are never attenuated; a
        warning is logged when they would have been. The weights, splittings
        and angles are stored in ``bloch`` (``edge_weights``, ``edge_split``,
        ``reactive_angle_deg``).
        """
        W = bloch["W"]
        split = bloch["split"]
        angle = reactive_angle_deg(W)
        weights = np.ones(W.shape, dtype=np.float64)
        edge = solver_config.BLOCH_EDGE_GUARD
        if edge is not None:
            weights *= edge_proximity_mode_weights(split, *edge)
        reactive = solver_config.BLOCH_EDGE_GUARD_DEG
        if forward_only and reactive is not None:
            weights *= edge_guard_mode_weights(W, *reactive)

        lead = tuple(range(W.ndim - 1))  # frequency (and segment) axes
        flagged_inj = injected & (weights < 1.0).any(axis=lead)
        if np.any(flagged_inj):
            cols = np.flatnonzero(flagged_inj)
            log.warning(
                "Bloch stopband-edge guard: injected mode(s) %s lie close to a "
                "stopband edge (min |sinh(gamma d)| = %s, max |arg W| = %s deg) "
                "and are kept; their description there is unreliable.",
                [mode_labels[j] for j in cols],
                [float(f"{split[..., j].min():.3g}") for j in cols],
                [round(float(angle[..., j].max()), 1) for j in cols],
            )
        weights[..., injected] = 1.0
        if keep is not None:
            weights[keep] = 1.0  # edge-basis modes: described exactly

        sig = np.asarray(signal_freqs, dtype=np.float64)
        for j in np.flatnonzero((weights < 1.0).any(axis=lead)):
            hit = weights[..., j] < 1.0
            f_hit = hit.reshape(len(sig), -1).any(axis=1)
            dropped = (weights[..., j] == 0.0).reshape(len(sig), -1).any(axis=1)
            log.warning(
                "Bloch stopband-edge guard: mode %s attenuated at %d of %d signal "
                "frequencies (%.4g-%.4g), dropped at %d (min |sinh(gamma d)| = "
                "%.3g, max |arg W| = %.1f deg).",
                mode_labels[j],
                int(f_hit.sum()),
                len(sig),
                sig[f_hit].min(),
                sig[f_hit].max(),
                int(dropped.sum()),
                float(split[..., j][hit].min()),
                float(angle[..., j][hit].max()),
            )
        bloch["edge_weights"] = weights
        bloch["edge_split"] = split
        bloch["reactive_angle_deg"] = angle
        return weights

    def _bloch_edge_guard(
        self,
        bloch: dict[str, Any],
        cme_inputs: dict[str, Any],
        signal_freqs: FloatArray,
        injected: np.ndarray,
        forward_only: bool = False,
        keep: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Stopband-edge guard of the uniform Bloch models.

        Per-mode weights from :meth:`_bloch_mode_weights`; returns the
        per-process weights (product over the modes of each relation row) for
        the 3WM and 4WM relation rows, shape (n_freq, n_rel).
        """
        weights = self._bloch_mode_weights(
            bloch, cme_inputs["mode_labels"], signal_freqs, injected, forward_only, keep
        )
        return (
            process_weights(cme_inputs["relations_3wm"], weights),
            process_weights(cme_inputs["relations_4wm"], weights),
        )

    def _prepare_bloch_coefficients(
        self,
        mode_array: ModeArray,
        cme_inputs: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Compute Bloch-basis coupling coefficients for a non-segmented device.

        Evaluated once per frequency point, after the mode frequencies are fixed:
        the coefficients depend only on the linear Bloch modes, not on the
        amplitudes or the position.

        Returns
        -------
        dict
            ``coeffs_3wm``, ``coeffs_4wm`` with shape (n_freq, n_relations), plus
            the Bloch tables used to build them (``W``, ``split``, ``Zb_plus``,
            ``Zb_minus``, ``phi_plus``, ``phi_minus``, ``L_nl``, ``n_cells``)
            and the diagnostics ``eig_residual`` and ``gamma_mismatch``
            (per-unit-cell |gamma_exact - gamma_interpolated|).
        """
        mode_labels = cme_inputs["mode_labels"]
        mode_params = cme_inputs["mode_params"]
        n_freq = cme_inputs["n_freq"]
        n_modes = cme_inputs["n_modes"]
        gammas = cme_inputs["gammas"]

        # Unit coefficients (RWA multiplicity / 4), same relation rows as cme_inputs.
        _, _, unit_3wm, unit_4wm = prepare_relations_coefficients(
            mode_array.get_rwa_terms(power=2),
            mode_array.get_rwa_terms(power=3),
            1.0,
            1.0,
        )

        freqs_hz = (
            np.stack([np.abs(mode_params[m]["freqs"]) for m in mode_labels], axis=-1)
            * self.unit_multiplier
        )
        # Kernel convention: gammas = -alpha - j k, so the propagation constant
        # of the forward Bloch mode is -gammas = alpha + j k.
        tables = bloch_mode_tables(self.twpa, freqs_hz.ravel(), -gammas.ravel())
        n_nl = tables["L_nl"].size
        shaped = {
            key: tables[key].reshape(n_freq, n_modes)
            for key in (
                "W",
                "Zb_plus",
                "Zb_minus",
                "gamma_exact",
                "eig_residual",
                "split",
            )
        }
        phi_plus = tables["phi_plus"].reshape(n_freq, n_modes, n_nl)
        phi_minus = tables["phi_minus"].reshape(n_freq, n_modes, n_nl)
        omegas = 2 * np.pi * freqs_hz

        coeff_args = (
            omegas,
            phi_plus,
            phi_minus,
            shaped["W"],
            tables["L_nl"],
            tables["n_cells"],
        )
        coeffs_3wm = bloch_coefficients(
            cme_inputs["relations_3wm"], unit_3wm, self.twpa.epsilon, *coeff_args
        )
        coeffs_4wm = bloch_coefficients(
            cme_inputs["relations_4wm"], unit_4wm, self.twpa.xi, *coeff_args
        )

        return {
            "coeffs_3wm": coeffs_3wm,
            "coeffs_4wm": coeffs_4wm,
            "W": shaped["W"],
            "split": shaped["split"],
            "Zb_plus": shaped["Zb_plus"],
            "Zb_minus": shaped["Zb_minus"],
            "phi_plus": phi_plus,
            "phi_minus": phi_minus,
            "L_nl": tables["L_nl"],
            "positions": tables["positions"],
            "n_cells": tables["n_cells"],
            "eig_residual": shaped["eig_residual"],
            "gamma_mismatch": np.abs(shaped["gamma_exact"] + gammas),
            # Inputs of the forward-backward kernel, which builds the
            # coefficients of every direction combination itself.
            "base_3wm": unit_3wm * self.twpa.epsilon,
            "base_4wm": unit_4wm * self.twpa.xi,
            "prefac": -2j * omegas / shaped["W"],
            "freqs_hz": freqs_hz,
            "gamma_exact": shaped["gamma_exact"],
        }

    def _apply_edge_basis(
        self,
        bloch: dict[str, Any],
        cme_inputs: dict[str, Any],
        weak: np.ndarray,
        injected: np.ndarray,
    ) -> Optional[dict[str, Any]]:
        """
        Fixed ("edge") Bloch basis for weak modes close to a stopband edge.

        Uniform devices, forward-backward model. Every weak mode without input
        whose splitting is below ``e = solver_config.BLOCH_EDGE_BASIS`` is
        expanded on the lossless Bloch pair at the reference frequency of the
        nearest edge, where the splitting equals the same ``e`` (passband
        side): the basis is anchored at the switch-over point. Its Bloch tables and ``prefac`` in
        ``bloch`` are overwritten in place, and the returned dict holds the
        kernel gammas (``-1j * kappa_e`` for flagged modes), the per-unit-cell
        generator ``K`` (``edge_K``), the mask (``edge_flag``) and diagnostics
        (also stored in ``bloch``). ``coeffs_3wm``/``coeffs_4wm`` are rebuilt
        from the new tables. Returns None when nothing is flagged.
        """
        cfg = solver_config.BLOCH_EDGE_BASIS
        if cfg is None:
            return None
        e_edge = float(cfg)
        flag = (bloch["split"] < e_edge) & weak[None, :] & ~injected[None, :]
        if not flag.any():
            return None
        gammas = cme_inputs["gammas"]
        n_freq, n_modes = flag.shape
        n_cells = bloch["n_cells"]
        twpa_ll = lossless_copy(self.twpa)
        ii, nn = np.nonzero(flag)
        f_mode = bloch["freqs_hz"][ii, nn]
        # Edge index: k d = m pi at the edge, m with the parity of the edge.
        kd_pi = -gammas.imag[ii, nn] * n_cells / np.pi
        m_edge = np.empty(len(f_mode), dtype=np.int64)

        f_ref = np.empty(len(f_mode))
        cache: list[tuple[float, float, float, int]] = []  # (lo, hi, f_ref, parity)
        for q, f in enumerate(f_mode):
            hit = [c for c in cache if c[0] <= f <= c[1]]
            if hit:
                f_ref[q], parity = hit[0][2], hit[0][3]
            else:
                f_ref[q], f_edge, parity = edge_reference_frequency(twpa_ll, f, e_edge)
                lo, hi = sorted((f_ref[q], 2 * f_edge - f_ref[q]))
                cache.append((lo, hi, f_ref[q], parity))
            odd = 0 if parity > 0 else 1
            m_edge[q] = 2 * int(np.round((kd_pi[q] - odd) / 2)) + odd
            if abs(kd_pi[q] - m_edge[q]) > 0.5:
                log.warning(
                    "Edge basis: mode %s at %.4f GHz has k d / pi = %.3f, far "
                    "from its edge index %d.",
                    cme_inputs["mode_labels"][nn[q]],
                    f / 1e9,
                    kd_pi[q],
                    m_edge[q],
                )
        tab = edge_basis_tables(self.twpa, twpa_ll, f_mode, f_ref, m_edge)

        for key in ("Zb_plus", "Zb_minus", "phi_plus", "phi_minus"):
            bloch[key][ii, nn] = tab[key]
        bloch["W"][ii, nn] = tab["W"]
        omegas = 2 * np.pi * bloch["freqs_hz"]
        bloch["prefac"] = -2j * omegas / bloch["W"]
        gammas_k = gammas.copy()
        gammas_k[ii, nn] = -1j * tab["kappa_e"]

        edge_K = np.zeros((n_freq, n_modes, 2, 2), dtype=np.complex128)
        edge_K[ii, nn] = tab["K"]
        L = float(self.twpa.N_tot)
        f_ref_tab = np.full((n_freq, n_modes), np.nan)
        f_ref_tab[ii, nn] = f_ref
        bragg = np.zeros((n_freq, n_modes))
        bragg[ii, nn] = np.abs(tab["K"][:, 0, 1]) * L
        ham = np.zeros((n_freq, n_modes))
        ham[ii, nn] = tab["ham_defect"]
        bloch.update(
            edge_flag=flag,
            edge_f_ref=f_ref_tab,
            edge_bragg_strength=bragg,
            edge_ham_defect=ham,
        )
        coeff_args = (
            omegas,
            bloch["phi_plus"],
            bloch["phi_minus"],
            bloch["W"],
            bloch["L_nl"],
            n_cells,
        )
        for key, rel in (("3wm", "relations_3wm"), ("4wm", "relations_4wm")):
            bloch[f"coeffs_{key}"] = bloch_coefficients(
                cme_inputs[rel], bloch[f"base_{key}"], 1.0, *coeff_args
            )

        labels = cme_inputs["mode_labels"]
        for n in np.unique(nn):
            sel = nn == n
            log.info(
                "Edge basis: mode %s at %d points (%.4f-%.4f GHz), f_ref = %s GHz, "
                "max Bragg strength |K01| L = %.1f.",
                labels[n],
                int(sel.sum()),
                f_mode[sel].min() / 1e9,
                f_mode[sel].max() / 1e9,
                np.unique(np.round(f_ref[sel] / 1e9, 5)).tolist(),
                float(bragg[ii[sel], n].max()),
            )
        return {"gammas": gammas_k, "edge_K": edge_K, "edge_flag": flag}

    def _prepare_bloch_segmented(
        self,
        mode_array: ModeArray,
        cme_inputs: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Per-segment Bloch tables of a segmented device for the forward-backward solver.

        Every segment is a uniform ``TWPA`` made of whole supercells; its Bloch
        modes are computed with :func:`bloch_mode_tables` on the segment's own
        supercell, at the mode frequencies, with the branch of ``k`` set by the
        segment's kernel wavenumber. Segments with identical cells share the
        computation. Intracell arrays are zero-padded to the largest number of
        nonlinear inductors per supercell.

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
        mode_labels = cme_inputs["mode_labels"]
        mode_params = cme_inputs["mode_params"]
        n_freq = cme_inputs["n_freq"]
        n_modes = cme_inputs["n_modes"]
        gammas = cme_inputs["gammas"]  # (n_freq, n_modes, n_seg)
        segments = [c for c in self.twpa.cells if isinstance(c, TWPA)]
        n_seg = len(segments)

        _, _, unit_3wm, unit_4wm = prepare_relations_coefficients(
            mode_array.get_rwa_terms(power=2),
            mode_array.get_rwa_terms(power=3),
            1.0,
            1.0,
        )
        freqs_hz = (
            np.stack([np.abs(mode_params[m]["freqs"]) for m in mode_labels], axis=-1)
            * self.unit_multiplier
        )
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
                raise ValueError(
                    f"Segment {s} does not hold a whole number of supercells."
                )
            per_seg.append(tab)
        log.info(
            "Bloch tables computed for %d distinct of %d segments.", len(cache), n_seg
        )

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
            "seg_bounds": np.asarray(self.data["segments"]["boundaries"], dtype=float),
            "eig_residual": eig_res,
            "gamma_mismatch": g_mis,
            "base_3wm": np.stack([unit_3wm * seg.epsilon for seg in segments]),
            "base_4wm": np.stack([unit_4wm * seg.xi for seg in segments]),
            "prefac": -2j * omegas[:, None, :] / W,
        }

    def _bloch_edge_guard_segmented(
        self,
        bloch: dict[str, Any],
        cme_inputs: dict[str, Any],
        signal_freqs: FloatArray,
        injected: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Stopband-edge guard of the segmented Bloch model.

        Same rule as :meth:`_bloch_edge_guard`, applied in every segment with
        that segment's Bloch modes: inside a segment, each process is scaled by
        the product of the weights of its modes there. Returns per-process
        weights of shape (n_freq, n_seg, n_rel).
        """
        weights = self._bloch_mode_weights(
            bloch, cme_inputs["mode_labels"], signal_freqs, injected, False
        )
        n_freq, n_seg, n_modes = weights.shape
        flat = weights.reshape(n_freq * n_seg, n_modes)
        pw3 = process_weights(cme_inputs["relations_3wm"], flat)
        pw4 = process_weights(cme_inputs["relations_4wm"], flat)
        return pw3.reshape(n_freq, n_seg, -1), pw4.reshape(n_freq, n_seg, -1)

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
        signal_arange: Optional[Tuple[float, float, float]] = None,
        pump_arange: Optional[Tuple[float, float, float]] = None,
        thin: int = 20,
        relative=True,
    ) -> dict[str, Any]:
        """Analyze phase matching conditions for nonlinear processes using mode arrays."""
        mode_array = self.get_mode_array(mode_array_config)
        freqs = self.data["freqs"]

        if pump_arange is not None:
            pump_f = (
                np.arange(*pump_arange)[::thin]
                if len(pump_arange) == 3
                else np.array(pump_arange)
            )
        else:
            min_p_idx = np.where(freqs == self.data["stopband_freqs"][1])[0][0]
            pump_f = freqs[min_p_idx:-1:thin]

        if signal_arange is not None:
            signal_f = (
                np.arange(*signal_arange)[::thin]
                if len(signal_arange) == 3
                else np.array(signal_arange)
            )
        else:
            max_s_idx = np.where(freqs == self.data["stopband_freqs"][0])[0][0]
            signal_f = freqs[:max_s_idx:thin]

        chi = pump_Ip0**2 * self.twpa.xi / 8 if pump_Ip0 else self.twpa.chi

        f_triplets = []
        k_triplets = []
        deltas = np.empty(shape=(len(signal_f), len(pump_f)))

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

            signal_freqs_pm = mode_params[signal_mode]["freqs"]
            idler_freqs_pm = mode_params[idler_mode]["freqs"]
            pump_k = mode_array.get_mode(pump_mode).k
            signal_k_array = mode_params[signal_mode]["k"]
            idler_k_array = mode_params[idler_mode]["k"]
            gamma = mode_array.get_mode(pump_mode).gamma

            for j, (s_freq, i_freq) in enumerate(zip(signal_freqs_pm, idler_freqs_pm)):
                f_triplets.append([p_freq, s_freq, i_freq])
                k_triplets.append([pump_k, signal_k_array[j], idler_k_array[j]])

            if (
                gamma is not None
                and pump_k is not None
                and signal_k_array is not None
                and idler_k_array is not None
            ):
                deltas[:, i] = (
                    pump_coeff * pump_k
                    + signal_sign * signal_k_array
                    + idler_sign * idler_k_array
                    + chi
                    * (1 + np.abs(gamma) ** 2)
                    * (
                        pump_k
                        + 2 * signal_sign * signal_k_array
                        + 2 * idler_sign * idler_k_array
                    )
                )
                if relative:
                    deltas[:, i] = deltas[:, i] / signal_k_array

        return {
            "delta": deltas,
            "triplets": {"f": f_triplets, "k": k_triplets},
            "pump_freqs": pump_f,
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
        pump: Optional[float] = None,
        Is0: float = 1e-7,
        Ip0: Optional[float] = None,
        model: Union[str, GainModel] = GainModel.GENERAL_NO_REFLECTIONS,
        mode_array_config: str = "basic_3wm",
        signal_mode: str = "s",
        pump_mode: str = "p",
        idler_mode: str = "i",
        initial_amplitudes_fwd: Optional[Union[List[float], np.ndarray]] = None,
        initial_amplitudes_bwd: Optional[Union[List[float], np.ndarray]] = None,
        thin: int = 100,
        fb_config: FBSolverConfig = FBSolverConfig(),
    ) -> dict[str, Any]:
        """
        Compute the TWPA gain using coupled mode equation models.

        Unified gain computation method supporting all CME models including
        the forward-backward model with reflection convergence.

        Parameters
        ----------
        signal_freqs : array_like
            Signal frequencies for gain calculation.
        pump : float, optional
            Pump frequency. If None, uses automatically determined optimal frequency.
        Is0 : float, default 1e-6
            Initial signal current amplitude (A).
        Ip0 : float, optional
            Initial pump current amplitude (A). If None, uses TWPA's Ip0 parameter.
        model : GainModel or str, default 'GENERAL_NO_REFLECTIONS'
            CME model:
            * 'general_no_reflections': Extended modes, no reflections.
            * 'general': Full forward-backward model with reflection convergence.
            * 'bloch_no_reflections': Extended modes, no reflections, Bloch-mode
              basis (non-segmented devices only).
            * 'bloch': Forward-backward model in the Bloch-mode basis with exact
              boundary conditions; segmented devices are solved per segment with
              exact junction conditions.
        mode_array_config : str, default 'basic_3wm'
            Name of mode array configuration to use.
        signal_mode, pump_mode, idler_mode : str
            Mode labels within the mode array.
        initial_amplitudes_fwd, initial_amplitudes_bwd : array_like, optional
            Amplitudes of the waves injected at x=0 (traveling along +x) and at
            x=L (traveling along -x); override Is0 and Ip0. Backward inputs
            require a forward-backward model ('general' or 'bloch'); the
            no-reflection models raise a ValueError if any is nonzero.
        thin : int, default 100
            Position array thinning factor.
        fb_config : FBSolverConfig, default FBSolverConfig()
            Convergence parameters for the forward-backward solver (GENERAL model
            only): ``passes``, ``kappa_cutoff``, ``update_rate``,
            ``convergence_threshold``, ``save_all_passes``.
        Returns
        -------
        dict
            Gain analysis results with model-specific fields. The
            forward-backward models (``general``, ``bloch``) also return the
            port-resolved signal gains ``gain_db`` (S21) and ``gain_db_11``
            (S11), referenced to the forward input, and ``gain_db_12`` (S12)
            and ``gain_db_22`` (S22), referenced to the backward input set with
            ``initial_amplitudes_bwd``. A gain whose reference input is zero is
            NaN. With inputs at both ends the outputs contain the response to
            both, so inject from one side at a time for S-parameters.
        """
        if isinstance(model, str):
            try:
                model = GainModel(model)
            except ValueError:
                raise ValueError(
                    f"Invalid model: {model}. Must be one of {[m.value for m in GainModel]}"
                )

        no_reflection_models = (
            GainModel.GENERAL_NO_REFLECTIONS,
            GainModel.BLOCH_NO_REFLECTIONS,
        )
        if (
            model in no_reflection_models
            and initial_amplitudes_bwd is not None
            and np.any(np.asarray(initial_amplitudes_bwd) != 0)
        ):
            raise ValueError(
                f"Model '{model.value}' propagates only +x traveling waves and "
                "cannot use backward inputs (initial_amplitudes_bwd). Use the "
                "forward-backward models ('general' or 'bloch') for "
                "counter-propagating waves."
            )

        if Ip0 is not None:
            self.twpa.Ip0 = Ip0
        if pump is None:
            pump = self.data["optimal_pump_freq"]

        signal_freqs = np.asarray(signal_freqs)
        N_tot = self.twpa.N_tot
        x = np.linspace(0, N_tot, int(N_tot / thin), endpoint=True, dtype=np.float64)

        if model == GainModel.GENERAL:
            return self._compute_forward_backward_gain(
                signal_freqs,
                pump,
                Is0,
                mode_array_config,
                signal_mode,
                pump_mode,
                idler_mode,
                x,
                initial_amplitudes_fwd,
                initial_amplitudes_bwd,
                fb_config,
            )
        elif model == GainModel.BLOCH:
            return self._compute_bloch_forward_backward_gain(
                signal_freqs,
                pump,
                Is0,
                mode_array_config,
                signal_mode,
                pump_mode,
                idler_mode,
                x,
                initial_amplitudes_fwd,
                initial_amplitudes_bwd,
                fb_config,
            )
        elif model == GainModel.BLOCH_NO_REFLECTIONS:
            return self._compute_bloch_gain_no_reflections(
                signal_freqs,
                pump,
                mode_array_config,
                signal_mode,
                pump_mode,
                idler_mode,
                x,
                initial_amplitudes_fwd,
                Is0,
            )
        else:
            return self._compute_general_gain_no_reflections(
                signal_freqs,
                pump,
                mode_array_config,
                signal_mode,
                pump_mode,
                idler_mode,
                x,
                model,
                initial_amplitudes_fwd,
                Is0,
            )

    # Private CME solvers

    def _compute_forward_backward_gain(
        self,
        signal_freqs: FloatArray,
        pump: float,
        Is0: float,
        mode_array_config: str,
        signal_mode: str,
        pump_mode: str,
        idler_mode: str,
        x: FloatArray,
        initial_amplitudes_fwd: Optional[Union[List[float], np.ndarray]] = None,
        initial_amplitudes_bwd: Optional[Union[List[float], np.ndarray]] = None,
        fb_config: Optional[FBSolverConfig] = None,
    ) -> dict[str, Any]:
        """Compute gain using the forward-backward propagation model."""
        fb_config = fb_config if fb_config is not None else FBSolverConfig()
        mode_array = self.get_mode_array(mode_array_config)

        # Forward wave: pump + signal, backward wave: zero (unless overridden).
        y0_fwd = self._setup_initial_conditions(
            mode_array,
            signal_mode,
            pump_mode,
            Is0,
            self.twpa.Ip0,
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
        gammas = cme_inputs["gammas"]
        Zb_table = cme_inputs["Zb_table"]
        mode_params = cme_inputs["mode_params"]

        # Build the per-boundary two-port tables (Zout_in/out + ABCD in/out).
        # When self.port/self.port_out are None these encode identity connectors
        # with Zout = self.twpa.Z0_ref, recovering the bare boundary.
        Zout_in, Zout_out, abcd_in, abcd_out = self._build_port_tables(
            mode_params, mode_labels, n_freq
        )

        y0_fwd_broadcast = self._broadcast_ic(y0_fwd, n_freq)
        y0_bwd_broadcast = self._broadcast_ic(y0_bwd, n_freq)

        I_tuples_raw = general_cmes_solve_fb_cutoff(
            x,
            y0_fwd_broadcast,
            y0_bwd_broadcast,
            gammas,
            Zb_table,
            cme_inputs["relations_3wm"],
            cme_inputs["relations_4wm"],
            cme_inputs["coeffs_3wm"],
            cme_inputs["coeffs_4wm"],
            fb_config.passes,
            fb_config.kappa_cutoff,
            update_rate=fb_config.update_rate,
            convergence_threshold=fb_config.convergence_threshold,
            save_all_passes=fb_config.save_all_passes,
            Z0=self.port_Z_ext,
            Ze=self.port_Z_ext,
            Zout_in=Zout_in,
            Zout_out=Zout_out,
            abcd_in=abcd_in,
            abcd_out=abcd_out,
            segment_midpoints=cme_inputs["segment_midpoints"],
        )

        # Unpack forward/backward blocks for each saved pass and apply phase correction.
        # Raw layout: [fwd_pass_0, ..., fwd_pass_N, bwd_pass_0, ..., bwd_pass_N]
        # each block is (n_freq, n_modes, n_x).
        num_saved_passes = fb_config.passes if fb_config.save_all_passes else 1
        I_tuples_all_passes = {}
        for p in range(num_saved_passes):
            if gammas.ndim == 3:
                gammas_bwd = gammas[:, :, ::-1]
            else:
                gammas_bwd = gammas
            I_tuples_all_passes[p] = {
                "forward": self._apply_phase_carrier(
                    I_tuples_raw[p * n_freq : (p + 1) * n_freq],
                    gammas,
                    x,
                    cme_inputs["segment_midpoints"],
                ),
                "backward": self._apply_phase_carrier(
                    I_tuples_raw[
                        num_saved_passes * n_freq
                        + p * n_freq : num_saved_passes * n_freq
                        + (p + 1) * n_freq
                    ],
                    gammas_bwd,
                    x,
                    cme_inputs["segment_midpoints"],
                ),
            }

        I_tuples_fwd = I_tuples_all_passes[num_saved_passes - 1]["forward"]
        I_tuples_bwd = I_tuples_all_passes[num_saved_passes - 1]["backward"]

        signal_idx = mode_labels.index(signal_mode)

        # Prompt boundary reflections Gamma_in/Gamma_out (Q2): the injected
        # drive reflecting directly off the near boundary without accounting
        # for propagation/generation of the counter-propagating wave along
        # the TWPA.  ABCD-correct form:
        #     Gamma = (Ze - Zin) / (Ze + Zin),   Zin = (A*Ztilde + B)/(C*Ztilde + D)
        # evaluated in the linear limit Ztilde ~= Zb (the nonlinear correction
        # is handled self-consistently by the solver's internal rho update).
        Ze = self.port_Z_ext
        Zb_in_sig = (
            Zb_table[:, signal_idx, 0]
            if Zb_table.ndim == 3
            else Zb_table[:, signal_idx]
        )
        Zb_out_sig = (
            Zb_table[:, signal_idx, -1]
            if Zb_table.ndim == 3
            else Zb_table[:, signal_idx]
        )

        A_in, B_in, C_in, D_in = (abcd_in[:, signal_idx, k] for k in range(4))
        Zin_in = (A_in * Zb_in_sig + B_in) / (C_in * Zb_in_sig + D_in)
        refl_in = (Ze - Zin_in) / (Ze + Zin_in)

        A_out, B_out, C_out, D_out = (abcd_out[:, signal_idx, k] for k in range(4))
        Zin_out = (A_out * Zb_out_sig + B_out) / (C_out * Zb_out_sig + D_out)
        refl_out = (Ze - Zin_out) / (Ze + Zin_out)

        I_out_fwd = (
            I_tuples_fwd[:, signal_idx, -1] + y0_bwd_broadcast[:, signal_idx] * refl_out
        )
        I_out_bwd = (
            I_tuples_bwd[:, signal_idx, -1] + y0_fwd_broadcast[:, signal_idx] * refl_in
        )

        Is0_rep = np.repeat(y0_fwd[signal_idx], n_freq)
        Ib0_rep = np.repeat(y0_bwd[signal_idx], n_freq)
        gains_fwd = _ratio_db(I_out_fwd, Is0_rep)
        gains_bwd = _ratio_db(I_out_bwd, Ib0_rep)
        gains_fwd_11 = _ratio_db(I_out_bwd, Is0_rep)
        gains_bwd_22 = _ratio_db(I_out_fwd, Ib0_rep)

        gains_all_passes = []
        for p in range(num_saved_passes):
            I_fwd_p = I_tuples_all_passes[p]["forward"]
            I_bwd_p = I_tuples_all_passes[p]["backward"]
            I_out_fwd_p = (
                I_fwd_p[:, signal_idx, -1] + y0_bwd_broadcast[:, signal_idx] * refl_out
            )
            I_out_bwd_p = (
                I_bwd_p[:, signal_idx, -1] + y0_fwd_broadcast[:, signal_idx] * refl_in
            )
            gains_all_passes.append(
                {
                    "gain_db_fwd": _ratio_db(I_out_fwd_p, Is0_rep),
                    "gain_db_bwd": _ratio_db(I_out_bwd_p, Ib0_rep),
                    "gain_db_11": _ratio_db(I_out_bwd_p, Is0_rep),
                    "gain_db_22": _ratio_db(I_out_fwd_p, Ib0_rep),
                }
            )

        mode_map = {label: idx for idx, label in enumerate(mode_labels)}

        return {
            "model": GainModel.GENERAL.value,
            "pump_freq": pump,
            "signal_freqs": signal_freqs,
            "x": x,
            "I_tuples": I_tuples_fwd,  # final-pass forward (backward-compat alias)
            "I_tuples_fwd": I_tuples_fwd,
            "I_tuples_bwd": I_tuples_bwd,
            "I_tuples_all_passes": I_tuples_all_passes,
            "gain_db": gains_fwd,
            "gain_db_12": gains_bwd,
            "gain_db_11": gains_fwd_11,
            "gain_db_22": gains_bwd_22,
            "gains_all_passes": gains_all_passes,
            "passes": fb_config.passes,
            "mode_info": {
                "mode_array": mode_array_config,
                "mode_map": mode_map,
                "reverse_mode_map": {idx: label for label, idx in mode_map.items()},
                "signal_mode": signal_mode,
                "pump_mode": pump_mode,
                "idler_mode": idler_mode,
            },
        }

    def _compute_general_gain_no_reflections(
        self,
        signal_freqs: FloatArray,
        pump: float,
        mode_array_config: str,
        signal_mode: str,
        pump_mode: str,
        idler_mode: str,
        x: FloatArray,
        model: GainModel,
        initial_amplitudes_fwd: Optional[Union[List[float], np.ndarray]] = None,
        Is0: float = 1e-6,
    ) -> dict[str, Any]:
        """Compute gain using the general coupled mode equations without reflections."""
        mode_array = self.get_mode_array(mode_array_config)

        y0 = self._setup_initial_conditions(
            mode_array,
            signal_mode,
            pump_mode,
            Is0,
            self.twpa.Ip0,
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
        )

        I_tuples_array = self._apply_phase_carrier(
            I_tuples_array, gammas, x, cme_inputs["segment_midpoints"]
        )

        signal_idx = mode_labels.index(signal_mode)
        initial_signal = y0[signal_idx] if y0.ndim == 1 else y0[:, signal_idx]
        gain_db = 10 * np.log10(
            np.abs(I_tuples_array[:, signal_idx, -1] / initial_signal) ** 2
        )

        mode_map = {label: idx for idx, label in enumerate(mode_labels)}

        return {
            "model": model.value,
            "pump_freq": pump,
            "signal_freqs": signal_freqs,
            "x": x,
            "I_tuples": I_tuples_array,
            "gain_db": gain_db,
            "mode_info": {
                "mode_array": mode_array_config,
                "mode_map": mode_map,
                "reverse_mode_map": {idx: label for label, idx in mode_map.items()},
                "signal_mode": signal_mode,
                "pump_mode": pump_mode,
                "idler_mode": idler_mode,
            },
        }

    def _compute_bloch_gain_no_reflections(
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
    ) -> dict[str, Any]:
        """
        Compute gain with the no-reflections CMEs written in the Bloch-mode basis.

        Amplitudes are the boundary currents of the forward Bloch modes at the
        supercell boundaries (x = 0 and x = N_tot are supercell boundaries), so
        initial conditions and outputs have the same meaning as in the
        standard model. See :mod:`twpasolver.cmes.bloch` for the theory.
        """
        if self._is_segmented():
            raise NotImplementedError(
                "The bloch_no_reflections model currently supports only "
                "non-segmented devices."
            )
        mode_array = self.get_mode_array(mode_array_config)

        y0 = self._setup_initial_conditions(
            mode_array,
            signal_mode,
            pump_mode,
            Is0,
            self.twpa.Ip0,
            initial_amplitudes_fwd,
        )

        cme_inputs = self._prepare_cme_inputs(
            mode_array, signal_freqs, pump, signal_mode, pump_mode
        )
        mode_labels = cme_inputs["mode_labels"]
        n_freq = cme_inputs["n_freq"]
        gammas = cme_inputs["gammas"]

        bloch = self._prepare_bloch_coefficients(mode_array, cme_inputs)

        # Keep only the terms described by cell-averaged couplings.
        kappas = gammas.imag
        coeffs_3wm = bloch["coeffs_3wm"] * cell_phase_mask(
            cme_inputs["relations_3wm"],
            kappas,
            bloch["n_cells"],
            solver_config.BLOCH_CELL_PHASE_CUTOFF,
        )
        coeffs_4wm = bloch["coeffs_4wm"] * cell_phase_mask(
            cme_inputs["relations_4wm"],
            kappas,
            bloch["n_cells"],
            solver_config.BLOCH_CELL_PHASE_CUTOFF,
        )
        y0_table = self._broadcast_ic(y0, n_freq)
        pw3, pw4 = self._bloch_edge_guard(
            bloch,
            cme_inputs,
            signal_freqs,
            np.any(np.abs(y0_table) > 0, axis=0),
            forward_only=True,
        )
        coeffs_3wm = coeffs_3wm * pw3
        coeffs_4wm = coeffs_4wm * pw4

        I_tuples_array = bloch_no_reflections_cmes_solve(
            x,
            y0_table,
            gammas,
            cme_inputs["relations_3wm"],
            cme_inputs["relations_4wm"],
            coeffs_3wm,
            coeffs_4wm,
        )

        I_tuples_array = self._apply_phase_carrier(I_tuples_array, gammas, x)

        signal_idx = mode_labels.index(signal_mode)
        initial_signal = y0[signal_idx] if y0.ndim == 1 else y0[:, signal_idx]
        gain_db = 10 * np.log10(
            np.abs(I_tuples_array[:, signal_idx, -1] / initial_signal) ** 2
        )

        mode_map = {label: idx for idx, label in enumerate(mode_labels)}

        return {
            "model": GainModel.BLOCH_NO_REFLECTIONS.value,
            "pump_freq": pump,
            "signal_freqs": signal_freqs,
            "x": x,
            "I_tuples": I_tuples_array,
            "gain_db": gain_db,
            "bloch": bloch,
            "mode_info": {
                "mode_array": mode_array_config,
                "mode_map": mode_map,
                "reverse_mode_map": {idx: label for label, idx in mode_map.items()},
                "signal_mode": signal_mode,
                "pump_mode": pump_mode,
                "idler_mode": idler_mode,
            },
        }

    def _compute_bloch_forward_backward_gain(
        self,
        signal_freqs: FloatArray,
        pump: float,
        Is0: float,
        mode_array_config: str,
        signal_mode: str,
        pump_mode: str,
        idler_mode: str,
        x: FloatArray,
        initial_amplitudes_fwd: Optional[Union[List[float], np.ndarray]] = None,
        initial_amplitudes_bwd: Optional[Union[List[float], np.ndarray]] = None,
        fb_config: Optional[FBSolverConfig] = None,
    ) -> dict[str, Any]:
        """
        Compute gain with the forward-backward CMEs in the Bloch-mode basis.

        Amplitudes are +x-referenced boundary currents of the forward and
        backward Bloch modes; boundary coefficients use the exact Bloch
        impedances Zb+ (forward) and Zb- (backward). ``fb_config.update_rate``
        is not used: the boundary coefficients are constant in this basis.
        See :mod:`twpasolver.cmes.bloch_fb`. Segmented devices (a ``TWPA`` whose
        cells are all ``TWPA`` segments) are solved on per-segment Bloch modes
        joined by exact junction conditions, see
        :mod:`twpasolver.cmes.bloch_fb_segmented`.
        """
        segmented = self._is_segmented()
        fb_config = fb_config if fb_config is not None else FBSolverConfig()
        mode_array = self.get_mode_array(mode_array_config)

        y0_fwd = self._setup_initial_conditions(
            mode_array,
            signal_mode,
            pump_mode,
            Is0,
            self.twpa.Ip0,
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
        gammas = cme_inputs["gammas"]
        mode_params = cme_inputs["mode_params"]
        if segmented:
            bloch = self._prepare_bloch_segmented(mode_array, cme_inputs)
        else:
            bloch = self._prepare_bloch_coefficients(mode_array, cme_inputs)

        Zout_in, Zout_out, abcd_in, abcd_out = self._build_port_tables(
            mode_params, mode_labels, n_freq
        )

        y0_fwd_broadcast = self._broadcast_ic(y0_fwd, n_freq)
        y0_bwd_broadcast = self._broadcast_ic(y0_bwd, n_freq)
        injected = np.any(np.abs(y0_fwd_broadcast) > 0, axis=0) | np.any(
            np.abs(y0_bwd_broadcast) > 0, axis=0
        )
        if segmented:
            pw3, pw4 = self._bloch_edge_guard_segmented(
                bloch, cme_inputs, signal_freqs, injected
            )
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
                passes=fb_config.passes,
                kappa_cutoff=fb_config.kappa_cutoff,
                m_max=fb_config.m_max,
                convergence_threshold=fb_config.convergence_threshold,
                save_all_passes=fb_config.save_all_passes,
                anderson_depth=fb_config.anderson_depth,
                process_weights_3wm=pw3,
                process_weights_4wm=pw4,
                Ze=self.port_Z_ext,
                Zout_in=Zout_in,
                Zout_out=Zout_out,
                abcd_in=abcd_in,
                abcd_out=abcd_out,
            )
            # Exact phase of the piecewise-constant wavenumber.
            Phi_f, Phi_b = segmented_phase(gammas.imag, bloch["seg_bounds"], x)
            carrier_fwd = np.exp(1j * Phi_f)
            carrier_bwd = np.exp(1j * Phi_b)
            # Launched waves: forward mode of the first segment at x=0,
            # backward mode of the last segment at x=L.
            Zp_launch = bloch["Zb_plus"][:, 0]
            Zm_launch = bloch["Zb_minus"][:, -1]
        else:
            weak_labels = set(mode_array.get_dependent_modes(signal_mode))
            weak_labels.add(signal_mode)
            weak = np.array([lab in weak_labels for lab in mode_labels])
            edge = self._apply_edge_basis(bloch, cme_inputs, weak, injected)
            diag: dict[str, Any] = {}
            if edge is not None:
                gammas = edge["gammas"]  # kernel and carrier: -1j kappa_e
            pw3, pw4 = self._bloch_edge_guard(
                bloch,
                cme_inputs,
                signal_freqs,
                injected,
                keep=None if edge is None else edge["edge_flag"],
            )
            I_tuples_raw = bloch_fb_cmes_solve(
                x,
                y0_fwd_broadcast,
                y0_bwd_broadcast,
                gammas,
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
                passes=fb_config.passes,
                kappa_cutoff=fb_config.kappa_cutoff,
                m_max=fb_config.m_max,
                convergence_threshold=fb_config.convergence_threshold,
                save_all_passes=fb_config.save_all_passes,
                anderson_depth=fb_config.anderson_depth,
                process_weights_3wm=pw3,
                process_weights_4wm=pw4,
                Ze=self.port_Z_ext,
                Zout_in=Zout_in,
                Zout_out=Zout_out,
                abcd_in=abcd_in,
                abcd_out=abcd_out,
                edge_K=None if edge is None else edge["edge_K"],
                edge_flag=None if edge is None else edge["edge_flag"],
                diagnostics=diag,
            )
            bloch.update(diag)
            carrier_fwd = np.exp(np.einsum("ij,k->ijk", 1j * gammas.imag, x))
            carrier_bwd = carrier_fwd
            Zp_launch = bloch["Zb_plus"]
            Zm_launch = bloch["Zb_minus"]

        num_saved_passes = fb_config.passes if fb_config.save_all_passes else 1
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

        Is0_rep = np.repeat(y0_fwd[signal_idx], n_freq)
        Ib0_rep = np.repeat(y0_bwd[signal_idx], n_freq)

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

        mode_map = {label: idx for idx, label in enumerate(mode_labels)}

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
            "passes": fb_config.passes,
            "bloch": bloch,
            "mode_info": {
                "mode_array": mode_array_config,
                "mode_map": mode_map,
                "reverse_mode_map": {idx: label for label, idx in mode_map.items()},
                "signal_mode": signal_mode,
                "pump_mode": pump_mode,
                "idler_mode": idler_mode,
            },
        }

    # Bandwidth

    @analysis_function
    def bandwidth(
        self,
        gain_reduction: float = 3,
        **gain_kwargs,
    ) -> dict[str, Any]:
        """
        Compute information about the compound gain bandwidth.

        Parameters
        ----------
        gain_reduction : float
            Difference from maximum gain used as threshold.
        **gain_kwargs : dict
            Additional arguments forwarded to ``gain()``.
        """
        if gain_kwargs or "gain" not in self.data.keys():
            self.gain(**gain_kwargs)

        gain_data = self.data["gain"]
        gain_db = gain_data["gain_db"]
        signal_freqs = gain_data["signal_freqs"]

        max_g = gain_db[np.argmax(gain_db)]
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
