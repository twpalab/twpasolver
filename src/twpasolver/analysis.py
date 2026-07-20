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
from twpasolver.cmes.cmes_fb import general_cmes_solve_fb_cutoff
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
    plot_quantum_efficiency,
    plot_response,
)
from twpasolver.twoport import TwoPortCell, TwoPortModel


class GainModel(str, enum.Enum):
    """
    Available CME models for gain analysis.

    Args
    ----
    GENERAL_NO_REFLECTIONS : str
        Extended mode system without reflections. Moderate computation
        time with improved accuracy over basic model.

    GENERAL : str
        Forward-backward propagation model with iterative solving for reflections.
        Most complete model including multiple passes for reflection convergence.
        Use phase mismatch kappa_cutoff limit to greatly increase computation speed.
    """

    GENERAL_NO_REFLECTIONS = "general_no_reflections"
    GENERAL = "general"


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
        results = {target: []}
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
            "Connector model at x=0 (input boundary).  Accepts either:\n"
            "  • A ``TwoPortModel`` instance (port 1 → external line, port 2 → TWPA).\n"
            "  • A precomputed list of impedances (one per frequency in\n"
            "    ``self.data['freqs']``), treated as a bare impedance step.\n"
            "When ``port_out`` is None this model is also used at x=L."
        ),
        validate=False,
    )
    port_out: Any = Field(
        default=None,
        description=(
            "Connector model at x=L (output boundary).  Same format as ``port``.\n"
            "When None the input connector model ``port`` is used at both boundaries,\n"
            "which is correct for identical connectors on both sides."
        ),
        validate=False,
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
                twpa = TWPA.from_file(twpa)
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

    def _bloch_params_for_cell(self, cell: TWPA, Z0_override: float | None = None) -> dict[str, Any]:
        """Extract Bloch-mode propagation parameters for a single uniform TWPA segment.

        Parameters
        ----------
        cell : TWPA
            The cell whose ABCD matrix is used to compute Bloch parameters.
        Z0_override : float, optional
            Reference impedance to use for the reflection coefficient and Bloch
            impedance computation.  When provided it overrides ``cell.Z0_ref``.
            This should be set to the parent TWPA's ``Z0_ref`` when computing
            per-segment parameters so that all segments share the same reference
            as the full device.
        """
        Z0 = Z0_override if Z0_override is not None else cell.Z0_ref
        abcd_single = cell.single_abcd(self.f)
        A, B, C, D = abcd_single.A, abcd_single.B, abcd_single.C, abcd_single.D
        ad = A + D
        freqs = self.f / self.unit_multiplier

        gamma_bloch = np.log((ad + np.sqrt(ad**2 - 4)) / 2)
        Zb = -2 * B / (A - D - np.sqrt((A + D) ** 2 - 4))
        gamma_bloch.real = np.abs(gamma_bloch.real) / int(cell.N_tot / cell.N)
        gamma_bloch.imag = (
            np.unwrap(gamma_bloch.imag * np.sign(Zb.real) * 2) / 2 / int(cell.N_tot / cell.N)
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
        Z0_in_table : (n_freq, n_modes) complex
            Zout = (D·Ze + B)/(C·Ze + A) at x=0.
        Z0_out_table : (n_freq, n_modes) complex
            Zout = (D·Ze + B)/(C·Ze + A) at x=L.
        abcd_in_table : (n_freq, n_modes, 4) complex
            [A,B,C,D] of input connector at x=0.
        abcd_out_table : (n_freq, n_modes, 4) complex
            [A,B,C,D] of output connector at x=L.

        When both port fields are None all tables encode identity connectors
        (A=1,B=0,C=0,D=1, Z0=self.twpa.Z0_ref), recovering bare behaviour.
        """
        n_modes = len(mode_labels)
        Ze = self.port_Z_ext

        Z0_default = np.full(
            (n_freq, n_modes), complex(self.twpa.Z0_ref), dtype=np.complex128
        )
        abcd_identity = np.zeros((n_freq, n_modes, 4), dtype=np.complex128)
        abcd_identity[:, :, 0] = 1.0   # A
        abcd_identity[:, :, 3] = 1.0   # D

        # Resolve effective port models: port_out falls back to port when None.
        port_in  = self.port
        port_out = self.port_out# if self.port_out is not None else self.port

        if port_in is None and port_out is None:
            return Z0_default, Z0_default.copy(), abcd_identity, abcd_identity.copy()

        Z0_in    = Z0_default.copy()
        Z0_out   = Z0_default.copy()
        abcd_in  = abcd_identity.copy()
        abcd_out = abcd_identity.copy()

        for m_idx, m in enumerate(mode_labels):
            freqs_m = np.abs(mode_params[m]["freqs"]) * self.unit_multiplier
            for table_Z0, table_abcd, port_model in (
                (Z0_in,  abcd_in,  port_in),
                (Z0_out, abcd_out, port_out),
            ):
                if port_model is None:
                    continue
                if isinstance(port_model, TwoPortModel):
                    abcd = np.asarray(port_model.get_abcd(freqs_m))  # (n_freq, 2, 2)
                    A, B = abcd[:, 0, 0], abcd[:, 0, 1]
                    C, D = abcd[:, 1, 0], abcd[:, 1, 1]
                    table_Z0[:, m_idx]      = (D * Ze + B) / (C * Ze + A)
                    table_abcd[:, m_idx, 0] = A; table_abcd[:, m_idx, 1] = B
                    table_abcd[:, m_idx, 2] = C; table_abcd[:, m_idx, 3] = D
                else:
                    table_Z0[:, m_idx] = np.interp(
                        freqs_m, self.data["freqs"] * self.unit_multiplier, port_model
                    )
       # print(Z0_in, Z0_out, abcd_in, abcd_out)
        return Z0_in, Z0_out, abcd_in, abcd_out

    def update_base_data(self) -> None:
        """Update data extracted from the twpa model."""
        current_state = self.twpa.model_dump()
        if (
            not hasattr(self, "_previous_state")
            or current_state != self._previous_state
        ):
            log.info("Computing base parameters.")
            abcd_single = self.twpa.single_abcd(self.f)
            cell = TwoPortCell(self.f, abcd=abcd_single**self.twpa.N, Z0=self.twpa.Z0_ref)
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
            if self._is_segmented():
                log.info(
                    "Detected %d TWPA segments, computing Bloch parameters per segment.",
                    len(self.twpa.cells),
                )
                seg_results = [
                    self._bloch_params_for_cell(seg, Z0_override=self.twpa.Z0_ref)
                    for seg in self.twpa.cells
                ]
                seg_N_tot = np.array([r["N_tot"] for r in seg_results], dtype=np.float64)
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
                    "epsilon": np.array([seg.epsilon for seg in self.twpa.cells]),
                    "xi": np.array([seg.xi for seg in self.twpa.cells]),
                }
                # k_star for response plots uses the whole-device Bloch k.
                k_full = self._bloch_params_for_cell(self.twpa)["k"]
            else:
                whole = self._bloch_params_for_cell(self.twpa)
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

    def _initialize_standard_mode_arrays(self) -> None:
        """Initialize standard mode arrays using the computed base data."""
        self._mode_arrays["basic_3wm"] = ModeArrayFactory.create_basic(self.data)
        self._mode_arrays["basic_4wm"] = ModeArrayFactory.create_basic(
            self.data, three_wave=False
        )

    def _update_all_mode_arrays(self) -> None:
        """Update all registered mode arrays with current base data."""
        if not self._mode_arrays:
            return
        interpolator = ParameterInterpolator(
            self.data["freqs"],
            {
                "k": self.data["k"],
                "gamma": self.data["reflection_coeff"],
                "alpha": self.data["alpha"],
                "Zb": self.data["Zb"],
            },
        )
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

    # ------------------------------------------------------------------
    # Shared CME helpers
    # ------------------------------------------------------------------

    def _setup_initial_conditions(
        self,
        mode_array: ModeArray,
        signal_mode: str,
        pump_mode: str,
        Is0: float,
        initial_amplitudes: Optional[Union[List[float], np.ndarray]] = None,
        default_fill: float = 0,
    ) -> np.ndarray:
        """
        Build a 1-D complex128 IC vector for CME solving.

        If *initial_amplitudes* is provided it is returned as-is (cast to
        complex128).  Otherwise a vector of *default_fill* is constructed with
        the pump amplitude set to ``twpa.Ip0`` and the signal to *Is0*.
        """
        if initial_amplitudes is not None:
            return np.array(initial_amplitudes, dtype=np.complex128)
        n_modes = len(mode_array.modes)
        mode_labels = list(mode_array.modes.keys())
        y0 = np.full(n_modes, default_fill, dtype=np.complex128)
        y0[mode_labels.index(pump_mode)] = self.twpa.Ip0
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
            trapezoid rule on the interpolated grid is exact for that interpolant —
            consistent with the phi_table computed inside the segmented Numba kernel.

        Parameters
        ----------
        I_array          : (n_freq, n_modes, n_x)
        gammas           : (n_freq, n_modes) or (n_freq, n_modes, n_seg)
        x                : (n_x,)
        segment_midpoints: (n_seg,), required when gammas.ndim == 3
        """
        if gammas.ndim == 3:
            # Interpolate κ_m onto the output grid: (n_freq, n_modes, n_x)
            kappa_at_x = np.array([
                [np.interp(x, segment_midpoints, gammas[i, j].imag)
                 for j in range(gammas.shape[1])]
                for i in range(gammas.shape[0])
            ])
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

        # The complex propagation and reflection arrays share the same list
        # comprehensions regardless of the device type; only the stacking
        # axis and final transpose differ between the two branches.
        complex_gammas = [
            -mode_params[m]["alpha"] + 1j * mode_params[m]["k"] for m in mode_labels
        ]
        gamma_vals = [mode_params[m]["gamma"] for m in mode_labels]
        Zb_vals    = [mode_params[m]["Zb"]    for m in mode_labels]

        segments = self.data.get("segments")
        if segments is not None:
            # Relations (topology) are segment-independent; only the scalar
            # nonlinearity coefficients (epsilon, xi) vary.  Call
            # prepare_relations_coefficients once with unit scaling to get the
            # index tables, then build per-segment coefficient arrays by scaling.
            relations_3wm, relations_4wm, c3_unit, c4_unit = (
                prepare_relations_coefficients(terms_3wm, terms_4wm, 1.0, 1.0)
            )
            coeffs_3wm = np.stack(
                [c3_unit * eps for eps in segments["epsilon"]], axis=-1
            )
            coeffs_4wm = np.stack(
                [c4_unit * xi for xi in segments["xi"]], axis=-1
            )
            # mode_params[m]["k"] is (n_segments, n_freq) from the 2-D interpolator;
            # final shapes: (n_freq, n_modes, n_segments)
            gammas      = np.stack(complex_gammas, axis=-2).transpose(2, 1, 0)
            reflections = np.stack(gamma_vals,     axis=-2).transpose(2, 1, 0)
            Zb_table    = np.stack(Zb_vals,        axis=-2).transpose(2, 1, 0)
            segment_midpoints = segments["midpoints"]
        else:
            relations_3wm, relations_4wm, coeffs_3wm, coeffs_4wm = (
                prepare_relations_coefficients(
                    terms_3wm, terms_4wm, self.twpa.epsilon, self.twpa.xi
                )
            )
            gammas      = np.stack(complex_gammas, axis=-1)
            reflections = np.stack(gamma_vals,     axis=-1)
            Zb_table    = np.stack(Zb_vals,        axis=-1)
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

    # ------------------------------------------------------------------
    # Public analysis methods
    # ------------------------------------------------------------------

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
            idler_freqs_pm  = mode_params[idler_mode]["freqs"]
            pump_k          = mode_array.get_mode(pump_mode).k
            signal_k_array  = mode_params[signal_mode]["k"]
            idler_k_array   = mode_params[idler_mode]["k"]
            gamma           = mode_array.get_mode(pump_mode).gamma

            for j, (s_freq, i_freq) in enumerate(zip(signal_freqs_pm, idler_freqs_pm)):
                f_triplets.append([p_freq, s_freq, i_freq])
                k_triplets.append([pump_k, signal_k_array[j], idler_k_array[j]])

            if all(
                param is not None
                for param in [gamma, pump_k, signal_k_array, idler_k_array]
            ):
                deltas[:, i] = (
                    pump_coeff * pump_k
                    + signal_sign * signal_k_array
                    + idler_sign  * idler_k_array
                    + chi
                    * (1 + np.abs(gamma) ** 2)
                    * (
                        pump_k
                        + 2 * signal_sign * signal_k_array
                        + 2 * idler_sign  * idler_k_array
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
        Is0: float = 1e-6,
        Ip0: Optional[float] = None,
        model: Union[str, GainModel] = GainModel.GENERAL_NO_REFLECTIONS,
        mode_array_config: str = "basic_3wm",
        signal_mode: str = "s",
        pump_mode: str = "p",
        idler_mode: str = "i",
        initial_amplitudes_fwd: Optional[Union[List[float], np.ndarray]] = None,
        initial_amplitudes_bwd: Optional[Union[List[float], np.ndarray]] = None,
        thin: int = 100,
        passes: int = 3,
        kappa_cutoff: float = 0.2,
        update_rate: float = 0.9,
        convergence_threshold: float = 0.05,
        save_all_passes: bool = False,
        floquet: bool = False,
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
        mode_array_config : str, default 'basic_3wm'
            Name of mode array configuration to use.
        signal_mode, pump_mode, idler_mode : str
            Mode labels within the mode array.
        initial_amplitudes_fwd, initial_amplitudes_bwd : array_like, optional
            Initial conditions for forward/backward waves; override Is0 and Ip0.
        thin : int, default 100
            Position array thinning factor.
        passes : int, default 3
            Number of forward-backward passes for convergence (GENERAL model only).
        save_all_passes : bool, default False
            Store intermediate results for every pass (GENERAL model only).
        Returns
        -------
        dict
            Gain analysis results with model-specific fields.
        """
        if isinstance(model, str):
            try:
                model = GainModel(model)
            except ValueError:
                raise ValueError(
                    f"Invalid model: {model}. Must be one of {[m.value for m in GainModel]}"
                )

        if Ip0 is not None:
            self.twpa.Ip0 = Ip0
        if pump is None:
            pump = self.data["optimal_pump_freq"]

        signal_freqs = np.asarray(signal_freqs)
        N_tot = self.twpa.N_tot
        x = np.linspace(0, N_tot, int(N_tot / thin), endpoint=True)

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
                passes,
                kappa_cutoff=kappa_cutoff,
                convergence_threshold=convergence_threshold,
                save_all_passes=save_all_passes,
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
                floquet,
            )

    # ------------------------------------------------------------------
    # Private CME solvers
    # ------------------------------------------------------------------

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
        passes: int = 3,
        kappa_cutoff: Optional[float] = None,
        convergence_threshold: float = 0.05,
        save_all_passes: bool = False,
    ) -> dict[str, Any]:
        """Compute gain using the forward-backward propagation model."""
        mode_array = self.get_mode_array(mode_array_config)

        # Forward wave: pump + signal, backward wave: zero (unless overridden).
        y0_fwd = self._setup_initial_conditions(
            mode_array, signal_mode, pump_mode, Is0,
            initial_amplitudes_fwd, default_fill=0.0,
        )
        y0_bwd = self._setup_initial_conditions(
            mode_array, signal_mode, pump_mode, Is0=0.0,
            initial_amplitudes=initial_amplitudes_bwd, default_fill=0.0,
        )

        cme_inputs = self._prepare_cme_inputs(
            mode_array, signal_freqs, pump, signal_mode, pump_mode
        )
        mode_labels  = cme_inputs["mode_labels"]
        n_freq       = cme_inputs["n_freq"]
        gammas       = cme_inputs["gammas"]
        Zb_table     = cme_inputs["Zb_table"]
        mode_params  = cme_inputs["mode_params"]

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
            passes,
            kappa_cutoff,
            convergence_threshold=convergence_threshold,
            save_all_passes=save_all_passes,
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
        num_saved_passes = passes if save_all_passes else 1
        I_tuples_all_passes = {}
        for p in range(num_saved_passes):
            if gammas.ndim == 3:
                gammas_bwd = gammas[:, :, ::-1]
            else:
                gammas_bwd = gammas
            I_tuples_all_passes[p] = {
                "forward": self._apply_phase_carrier(
                    I_tuples_raw[p * n_freq:(p + 1) * n_freq],
                    gammas, x, cme_inputs["segment_midpoints"],
                ),
                "backward": self._apply_phase_carrier(
                    I_tuples_raw[
                        num_saved_passes * n_freq + p * n_freq:
                        num_saved_passes * n_freq + (p + 1) * n_freq
                    ],
                    gammas_bwd, x, cme_inputs["segment_midpoints"],
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
        Zb_in_sig  = Zb_table[:, signal_idx, 0]  if Zb_table.ndim == 3 else Zb_table[:, signal_idx]
        Zb_out_sig = Zb_table[:, signal_idx, -1] if Zb_table.ndim == 3 else Zb_table[:, signal_idx]

        A_in, B_in, C_in, D_in = (abcd_in[:, signal_idx, k] for k in range(4))
        Zin_in  = (A_in * Zb_in_sig + B_in) / (C_in * Zb_in_sig + D_in)
        refl_in = (Ze - Zin_in) / (Ze + Zin_in)

        A_out, B_out, C_out, D_out = (abcd_out[:, signal_idx, k] for k in range(4))
        Zin_out  = (A_out * Zb_out_sig + B_out) / (C_out * Zb_out_sig + D_out)
        refl_out = (Ze - Zin_out) / (Ze + Zin_out)

        I_out_fwd = (
            I_tuples_fwd[:, signal_idx, -1]
            + y0_bwd_broadcast[:, signal_idx] * refl_out
        )
        I_out_bwd = (
            I_tuples_bwd[:, signal_idx, -1]
            + y0_fwd_broadcast[:, signal_idx] * refl_in
        )

        Is0_rep = np.repeat(y0_fwd[signal_idx], n_freq)
        gains_fwd    = to_dB(np.abs(I_out_fwd / Is0_rep))
        gains_bwd    = to_dB(np.abs(I_out_bwd / np.repeat(y0_bwd[signal_idx], n_freq)))
        gains_fwd_11 = to_dB(np.abs(I_out_bwd / Is0_rep))
        gains_bwd_22 = to_dB(np.abs(I_out_fwd / np.repeat(y0_bwd[signal_idx], n_freq)))

        gains_all_passes = []
        for p in range(num_saved_passes):
            I_fwd_p = I_tuples_all_passes[p]["forward"]
            I_bwd_p = I_tuples_all_passes[p]["backward"]
            I_out_fwd_p = I_fwd_p[:, signal_idx, -1] + y0_bwd_broadcast[:, signal_idx] * refl_out
            I_out_bwd_p = I_bwd_p[:, signal_idx, -1] + y0_fwd_broadcast[:, signal_idx] * refl_in
            gains_all_passes.append({
                "gain_db_fwd": to_dB(np.abs(I_out_fwd_p / Is0_rep)),
                "gain_db_bwd": to_dB(np.abs(I_out_bwd_p / np.repeat(y0_bwd[signal_idx], n_freq))),
                "gain_db_11":  to_dB(np.abs(I_out_bwd_p / Is0_rep)),
                "gain_db_22":  to_dB(np.abs(I_out_fwd_p / np.repeat(y0_bwd[signal_idx], n_freq))),
            })

        mode_map = {label: idx for idx, label in enumerate(mode_labels)}

        return {
            "model": GainModel.GENERAL.value,
            "pump_freq": pump,
            "signal_freqs": signal_freqs,
            "x": x,
            "I_tuples": I_tuples_fwd,           # final-pass forward (backward-compat alias)
            "I_tuples_fwd": I_tuples_fwd,
            "I_tuples_bwd": I_tuples_bwd,
            "I_tuples_all_passes": I_tuples_all_passes,
            "gain_db": gains_fwd,
            "gain_db_12": gains_bwd,
            "gain_db_11": gains_fwd_11,
            "gain_db_22": gains_bwd_22,
            "gains_all_passes": gains_all_passes,
            "passes": passes,
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
        second_order: bool = False,
    ) -> dict[str, Any]:
        """Compute gain using the general coupled mode equations without reflections."""
        mode_array = self.get_mode_array(mode_array_config)

        y0 = self._setup_initial_conditions(
            mode_array, signal_mode, pump_mode, Is0, initial_amplitudes_fwd
        )

        cme_inputs = self._prepare_cme_inputs(
            mode_array, signal_freqs, pump, signal_mode, pump_mode
        )
        mode_labels = cme_inputs["mode_labels"]
        n_freq      = cme_inputs["n_freq"]
        gammas      = cme_inputs["gammas"]

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

    # ------------------------------------------------------------------
    # Bandwidth
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Plotting helpers
    # ------------------------------------------------------------------

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

    def plot_quantum_efficiency(self, **kwargs) -> np.ndarray:
        """Plot quantum efficiency."""
        if "quantum_efficiency" not in self.data:
            raise RuntimeError("Run quantum_efficiency() first.")
        return plot_quantum_efficiency(
            self.data["quantum_efficiency"],
            freqs_unit=self.unit,
            **kwargs,
        )