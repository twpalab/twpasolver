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
from twpasolver.twoport import TwoPortCell


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
        List of 3-wave mixing terms, each containing mode indices and coefficient
    terms_4wm : list
        List of 4-wave mixing terms, each containing mode indices and coefficient
    epsilon : float
        Scaling factor for 3-wave mixing coefficients
    xi : float
        Scaling factor for 4-wave mixing coefficients

    Returns
    -------
    relations_3wm : List of mode indices for 3-wave mixing
    relations_4wm : List of mode indices for 4-wave mixing
    coeffs_3wm : numpy array of scaled coefficients for 3-wave mixing
    coeffs_4wm : numpy array of scaled coefficients for 4-wave mixing
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

    relations_3wm = np.array(relations_3wm, dtype=np.int64)
    relations_4wm = np.array(relations_4wm, dtype=np.int64)
    coeffs_3wm = np.array(coeffs_3wm, dtype=np.complex128)
    coeffs_4wm = np.array(coeffs_4wm, dtype=np.complex128)

    return relations_3wm, relations_4wm, coeffs_3wm, coeffs_4wm


def analysis_function(func):
    """Decorate analysis methods providing automatic result caching and management."""

    @wraps(func)
    def wrapper(self, *args, save=True, save_name: Optional[str] = None, **kwargs):
        self.update_base_data()
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

    use_block: bool = True
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
                        "Results dictionary contains key with target variable name, will be saved as %s.",
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

    def update_base_data(self) -> None:
        """Update data extracted from the twpa model with clearer notation."""
        current_state = self.twpa.model_dump()
        if (
            not hasattr(self, "_previous_state")
            or current_state != self._previous_state
        ):
            log.info("Computing base parameters.")
            abcd_single = self.twpa.single_abcd(self.f)
            abcd_full = abcd_single**self.twpa.N
            cell = TwoPortCell(self.f, abcd=abcd_full, Z0=self.twpa.Z0_ref)
            freqs = self.f / self.unit_multiplier

            # Basic frequency and S-parameter data
            self.data["freqs"] = freqs
            self.data["abcd"] = np.asarray(cell.abcd)
            self.data["S21"] = cell.s.S21
            self.data["S11"] = cell.s.S11
            s21_db = 20 * np.log10(np.abs(self.data["S21"]))
            self.data["S21_db"] = s21_db

            # Stopband identification
            s21_db_diff = s21_db[1:] - s21_db[:-1]
            stopband_start_idx = np.argmin(s21_db_diff)
            self.data["stopband_freqs"] = [
                freqs[stopband_start_idx],
                freqs[np.argmax(s21_db_diff)],
            ]

            # Extract propagation parameters with clearer notation
            Z0 = self.twpa.Z0_ref
            A, B, C, D = (
                abcd_single.A,
                abcd_single.B,
                abcd_single.C,
                abcd_single.D,
            )
            ad = A + D
            n_current = self.twpa.N
            self.twpa.N = 1

            # Propagation constant (gamma_b in block model)
            gamma_bloch = np.log((ad + np.sqrt(ad**2 - 4)) / 2)
            Zb = -2 * B / (A - D - np.sqrt((A + D) ** 2 - 4))
            gamma_bloch.real = np.abs(gamma_bloch.real) / self.twpa.N_tot
            gamma_bloch.imag = (
                np.unwrap(gamma_bloch.imag * np.sign(Zb.real) * 2) / 2 / self.twpa.N_tot
            )
            Zb.real = np.abs(Zb.real)

            # Store with clearer names
            self.data["Zb"] = Zb
            self.data["gamma_bloch"] = gamma_bloch  # Renamed from gammab
            self.twpa.N = n_current

            # Compute attenuation and reflection coefficients
            N_tot = self.twpa.N_tot
            alpha = -np.log(np.abs(self.data["S21"])) / (N_tot)

            # Reflection coefficient (S11)
            self.data["reflection_coeff"] = (Zb - self.twpa.Z0_ref) / (
                Zb + self.twpa.Z0_ref
            )  # Renamed from gammas

            # Phase response and wavenumber
            if self.use_block:
                k_full = gamma_bloch.imag
                self.data["alpha"] = gamma_bloch.real
            else:
                k_full = -np.unwrap(np.angle(self.data["S21"])) / self.twpa.N_tot
                self.data["alpha"] = alpha

            # Wavenumber processing
            freqs_mask = freqs < freqs[stopband_start_idx] / 10
            if len(freqs_mask) > 2:
                k_pars = np.polyfit(freqs[freqs_mask], k_full[freqs_mask], 2)
                self.data["k"] = k_full - k_pars[-1]
                self.data["k_star"] = k_full - np.polyval(k_pars[1:], freqs)
            else:
                self.data["k"] = k_full
                self.data["k_star"] = k_full

            self.data["optimal_pump_freq"] = self._estimate_optimal_pump()
            self._previous_state = current_state

            # Update all registered mode arrays with new base data
            self._update_all_mode_arrays()

            # Initialize standard mode arrays if they don't exist
            if "basic_3wm" not in self._mode_arrays:
                self._initialize_standard_mode_arrays()

    def _initialize_standard_mode_arrays(self) -> None:
        """Initialize standard mode arrays using the computed base data."""
        self._mode_arrays["basic_3wm"] = ModeArrayFactory.create_basic(self.data)
        self._mode_arrays["basic_4wm"] = ModeArrayFactory.create_basic(
            self.data, three_wave=False
        )

    def _update_all_mode_arrays(self) -> None:
        """Update all registered mode arrays with current base data using clearer notation."""
        if not self._mode_arrays:
            return

        if self.use_block:
            interpolator = ParameterInterpolator(
                self.data["freqs"],
                self.data["gamma_bloch"].imag,  # kappa
                self.data["reflection_coeff"],  # gamma (reflection)
                self.data["gamma_bloch"].real,  # alpha
            )
        else:
            interpolator = ParameterInterpolator(
                self.data["freqs"],
                self.data["k"],  # kappa
                self.data["reflection_coeff"],  # gamma (reflection)
                self.data["alpha"],  # alpha
            )

        # Update all mode arrays
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
        ks = self.data["k"]
        min_p_idx = np.where(freqs == self.data["stopband_freqs"][1])[0][0]
        pump_f = freqs[min_p_idx:-1]
        pump_k = ks[min_p_idx:-1]
        signal_f = pump_f / 2
        signal_k = np.interp(signal_f, freqs, ks)
        deltas = np.abs(
            pump_k - 2 * signal_k + 1 * self.twpa.chi * (pump_k - 4 * signal_k)
        )
        return pump_f[np.argmin(deltas)]

    def _setup_initial_conditions(
        self,
        mode_array: ModeArray,
        signal_mode: str,
        pump_mode: str,
        Is0: float,
        initial_amplitudes_fwd: Optional[Union[List[float], np.ndarray]] = None,
    ) -> np.ndarray:
        """Set initial conditions for CME solving."""
        if initial_amplitudes_fwd is None:
            n_modes = len(mode_array.modes)
            y0 = np.full(n_modes, 1e-14, dtype=np.complex128)
            mode_labels = list(mode_array.modes.keys())
            signal_idx = mode_labels.index(signal_mode)
            pump_idx = mode_labels.index(pump_mode)
            y0[pump_idx] = self.twpa.Ip0
            y0[signal_idx] = Is0
        else:
            y0 = np.array(initial_amplitudes_fwd, dtype=np.complex128)
        return y0

    def _prepare_mode_parameters(
        self,
        mode_array: ModeArray,
        signal_freqs: FloatArray,
        pump: float,
        signal_mode: str,
    ) -> Tuple[dict, list]:
        """Prepare mode parameters for CME solving."""
        # Update pump frequency and process signal frequency array
        mode_array.update_frequencies(
            {signal_mode.replace("signal_mode", "pump_mode"): pump}
        )
        mode_params = mode_array.process_frequency_array(signal_mode, signal_freqs)

        # Get mode labels
        mode_labels = list(mode_array.modes.keys())

        return mode_params, mode_labels

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
        # Get the mode array
        mode_array = self.get_mode_array(mode_array_config)
        freqs = self.data["freqs"]

        if pump_arange is not None:
            if len(pump_arange) == 3:
                pump_f = np.arange(*pump_arange)[::thin]
            else:
                pump_f = np.array(pump_arange)
        else:
            min_p_idx = np.where(freqs == self.data["stopband_freqs"][1])[0][0]
            max_p_idx = -1
            pump_f = freqs[min_p_idx:max_p_idx:thin]

        if signal_arange is not None:
            if len(signal_arange) == 3:
                signal_f = np.arange(*signal_arange)[::thin]
            else:
                signal_f = np.array(signal_arange)
        else:
            max_s_idx = np.where(freqs == self.data["stopband_freqs"][0])[0][0]
            signal_f = freqs[:max_s_idx:thin]

        chi = pump_Ip0**2 * self.twpa.xi / 8 if pump_Ip0 else self.twpa.chi

        f_triplets = []
        k_triplets = []
        deltas = np.empty(shape=(len(signal_f), len(pump_f)))

        # Process type logic
        if process == "PA":
            idler_sign = -1
            signal_sign = -1
        elif process == "FCD":
            idler_sign = 1
            signal_sign = -1
        else:
            idler_sign = -1
            signal_sign = 1

        pump_coeff = 1
        if mixing == "4WM":
            pump_coeff = 2

        mode_array_keys = list(mode_array.modes.keys())
        signal_idx = mode_array_keys.index(signal_mode)
        pump_idx = mode_array_keys.index(pump_mode)
        idler_idx = mode_array_keys.index(idler_mode)

        # Compute phase matching
        for i, p_freq in enumerate(pump_f):
            mode_array.update_frequencies({pump_mode: p_freq})
            mode_params = mode_array.process_frequency_array(signal_mode, signal_f)

            signal_freqs = mode_params[signal_mode]["freqs"]
            idler_freqs = mode_params[idler_mode]["freqs"]

            pump_k = mode_array.get_mode(pump_mode).k
            signal_k_array = mode_params[signal_mode]["k"]
            idler_k_array = mode_params[idler_mode]["k"]
            gamma = mode_array.get_mode(pump_mode).gamma

            for j, (s_freq, i_freq) in enumerate(zip(signal_freqs, idler_freqs)):
                f_triplets.append([p_freq, s_freq, i_freq])
                k_triplets.append([pump_k, signal_k_array[j], idler_k_array[j]])

            if all(
                param is not None
                for param in [gamma, pump_k, signal_k_array, idler_k_array]
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
    ) -> dict[str, Any]:
        """
        Compute the TWPA gain using coupled mode equation models.

        Unified gain computation method supporting all CME models including forward-backward.

        Parameters
        ----------
        signal_freqs : array_like
            Signal frequencies for gain calculation
        pump : float, optional
            Pump frequency. If None, uses automatically determined optimal frequency
        Is0 : float, default 1e-6
            Initial signal current amplitude (A) for standard models
        Ip0 : float, optional
            Initial pump current amplitude (A). If None, uses TWPA's Ip0 parameter
        model : GainModel or str, default 'GENERAL_NO_REFLECTIONS'
            CME model complexity:
            * 'GENERAL_NO_REFLECTIONS': Basic pump-signal-idler only (fastest)
            * 'no_reflections': Extended modes, no losses/reflections
            * 'general': Full model with losses and reflections, uses iterative solving
        mode_array_config : str, default 'basic_3wm'
            Name of mode array configuration to use
        signal_mode, pump_mode, idler_mode : str
            Mode labels within the mode array
        initial_amplitudes_fwd, initial_amplitudes_bwd : array_like, optional
            Initial conditions for forward and backward waves, overrides Is0 and Ip0
        thin : int, default 100
            Position array thinning factor
        passes : int, default 3
            Number of forward-backward passes for convergence

        Returns
        -------
        dict
            Gain analysis results with model-specific fields
        """
        # Process model parameter
        if isinstance(model, str):
            try:
                model = GainModel(model)
            except ValueError:
                raise ValueError(
                    f"Invalid model: {model}. Must be one of {[m.value for m in GainModel]}"
                )

        # Set pump current if specified
        if Ip0 is not None:
            self.twpa.Ip0 = Ip0

        # Use optimal pump if not specified
        if pump is None:
            pump = self.data["optimal_pump_freq"]

        # Ensure signal_freqs is a numpy array
        signal_freqs = np.asarray(signal_freqs)

        # Get position array
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
        # Get the mode array
        mode_array = self.get_mode_array(mode_array_config)
        mode_labels = list(mode_array.modes.keys())

        # Setup initial conditions
        if initial_amplitudes_fwd is None:
            n_modes = len(mode_array.modes)
            y0_fwd = np.full(n_modes, 0, dtype=np.complex128)
            signal_idx = mode_labels.index(signal_mode)
            pump_idx = mode_labels.index(pump_mode)
            y0_fwd[pump_idx] = self.twpa.Ip0
            y0_fwd[signal_idx] = Is0
        else:
            y0_fwd = np.array(initial_amplitudes_fwd, dtype=np.complex128)

        if initial_amplitudes_bwd is None:
            y0_bwd = np.full(len(mode_array.modes), 0, dtype=np.complex128)
        else:
            y0_bwd = np.array(initial_amplitudes_bwd, dtype=np.complex128)

        # Prepare mode parameters
        N_tot = self.twpa.N_tot
        n_freq = len(signal_freqs)
        n_modes = len(mode_array.modes)

        mode_array.update_frequencies({pump_mode: pump})
        mode_params = mode_array.process_frequency_array(signal_mode, signal_freqs)

        # Get RWA terms
        terms_3wm = mode_array.get_rwa_terms(power=2)
        terms_4wm = mode_array.get_rwa_terms(power=3)
        (
            relations_3wm,
            relations_4wm,
            coeffs_3wm,
            coeffs_4wm,
        ) = prepare_relations_coefficients(
            terms_3wm, terms_4wm, self.twpa.epsilon, self.twpa.xi
        )

        # Build CME data arrays
        gammas = np.empty((n_freq, n_modes), dtype=np.complex128)
        reflections = np.empty((n_freq, n_modes), dtype=np.complex128)
        for i in range(n_freq):
            kappas = np.array([mode_params[mode]["k"][i] for mode in mode_labels])
            alphas = np.array([mode_params[mode]["alpha"][i] for mode in mode_labels])
            reflections[i] = np.array(
                [mode_params[mode]["gamma"][i] for mode in mode_labels]
            )
            gammas[i] = -alphas + 1j * kappas

        I_tuples = general_cmes_solve_fb_cutoff(
            x,
            y0_fwd,
            y0_bwd,
            gammas,
            reflections,
            relations_3wm,
            relations_4wm,
            coeffs_3wm,
            coeffs_4wm,
            passes,
            kappa_cutoff,
            conv_threshold=convergence_threshold,
            save_all_passes=save_all_passes,
        )

        # Extract results for all passes
        # I_tuples now has shape (2 * n_freq * passes, n_modes, len(x))
        I_tuples_all_passes = {}
        if save_all_passes:
            num_saved_passes = passes
        else:
            num_saved_passes = 1
        for pass_idx in range(num_saved_passes):
            # Forward results for this pass
            start_fwd = pass_idx * n_freq
            end_fwd = (pass_idx + 1) * n_freq
            I_tuples_fwd_pass = I_tuples[start_fwd:end_fwd]

            # Backward results for this pass
            start_bwd = n_freq * num_saved_passes + pass_idx * n_freq
            end_bwd = n_freq * num_saved_passes + (pass_idx + 1) * n_freq
            I_tuples_bwd_pass = I_tuples[start_bwd:end_bwd]

            # # Apply propagation phase correction
            I_tuples_fwd_pass = I_tuples_fwd_pass * np.exp(
                np.einsum("ij,k->ijk", gammas, x)
            )
            I_tuples_bwd_pass = I_tuples_bwd_pass * np.exp(
                np.einsum("ij,k->ijk", gammas, x)
            )

            I_tuples_all_passes[pass_idx] = {
                "forward": I_tuples_fwd_pass,
                "backward": I_tuples_bwd_pass,
            }

        # Use final pass results for main calculation
        I_tuples_fwd = I_tuples_all_passes[num_saved_passes - 1]["forward"]
        I_tuples_bwd = I_tuples_all_passes[num_saved_passes - 1]["backward"]

        # Calculate transmission coefficients and output currents
        transmission_coeffs = 1 + reflections
        signal_idx = mode_labels.index(signal_mode)

        I_out_fwd = I_tuples_fwd[:, signal_idx, -1] * transmission_coeffs[:, signal_idx]
        I_out_bwd = I_tuples_bwd[:, signal_idx, -1] * transmission_coeffs[:, signal_idx]

        # Calculate gains for final pass
        gains_fwd = to_dB(np.abs(I_out_fwd / np.repeat(y0_fwd[signal_idx], n_freq)))
        gains_bwd = to_dB(np.abs(I_out_bwd / np.repeat(y0_bwd[signal_idx], n_freq)))
        gains_fwd_11 = to_dB(np.abs(I_out_bwd / np.repeat(y0_fwd[signal_idx], n_freq)))
        gains_bwd_22 = to_dB(np.abs(I_out_fwd / np.repeat(y0_bwd[signal_idx], n_freq)))

        # Calculate gains for all num_saved_passes
        gains_all_passes = []
        for pass_idx in range(num_saved_passes):
            I_fwd_pass = I_tuples_all_passes[pass_idx]["forward"]
            I_bwd_pass = I_tuples_all_passes[pass_idx]["backward"]

            I_out_fwd_pass = (
                I_fwd_pass[:, signal_idx, -1] * transmission_coeffs[:, signal_idx]
            )
            I_out_bwd_pass = (
                I_bwd_pass[:, signal_idx, -1] * transmission_coeffs[:, signal_idx]
            )

            gains_all_passes.append(
                {
                    "gain_db_fwd": to_dB(
                        np.abs(I_out_fwd_pass / np.repeat(y0_fwd[signal_idx], n_freq))
                    ),
                    "gain_db_bwd": to_dB(
                        np.abs(I_out_bwd_pass / np.repeat(y0_bwd[signal_idx], n_freq))
                    ),
                    "gain_db_11": to_dB(
                        np.abs(I_out_bwd_pass / np.repeat(y0_fwd[signal_idx], n_freq))
                    ),
                    "gain_db_22": to_dB(
                        np.abs(I_out_fwd_pass / np.repeat(y0_bwd[signal_idx], n_freq))
                    ),
                }
            )

        # Create mode maps
        mode_map = {label: idx for idx, label in enumerate(mode_labels)}
        reverse_mode_map = {idx: label for label, idx in mode_map.items()}

        return {
            "model": GainModel.GENERAL.value,
            "pump_freq": pump,
            "signal_freqs": signal_freqs,
            "x": x,
            "I_tuples": I_tuples_fwd,  # Final pass forward results for backward compatibility
            "I_tuples_fwd": I_tuples_fwd,  # Final pass forward results
            "I_tuples_bwd": I_tuples_bwd,  # Final pass backward results
            "I_tuples_all_passes": I_tuples_all_passes,  # All num_saved_passes results
            "gain_db": gains_fwd,  # Final pass forward gain
            "gain_db_12": gains_bwd,  # Final pass backward gain
            "gain_db_11": gains_fwd_11,  # Final pass cross-coupling
            "gain_db_22": gains_bwd_22,  # Final pass cross-coupling
            "gains_all_passes": gains_all_passes,  # Gains for all passes
            "passes": passes,
            "mode_info": {
                "mode_array": mode_array_config,
                "mode_map": mode_map,
                "reverse_mode_map": reverse_mode_map,
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
        second_order=False,
    ) -> dict[str, Any]:
        """Compute gain using the general coupled mode equations models."""
        # Get the mode array
        mode_array = self.get_mode_array(mode_array_config)
        mode_labels = list(mode_array.modes.keys())

        # Setup initial conditions
        y0 = self._setup_initial_conditions(
            mode_array, signal_mode, pump_mode, Is0, initial_amplitudes_fwd
        )

        # Prepare mode parameters
        N_tot = self.twpa.N_tot
        n_freq = len(signal_freqs)
        n_modes = len(mode_array.modes)

        mode_array.update_frequencies({pump_mode: pump})
        mode_params = mode_array.process_frequency_array(signal_mode, signal_freqs)

        # Get RWA terms
        terms_3wm = mode_array.get_rwa_terms(power=2)
        terms_4wm = mode_array.get_rwa_terms(power=3)
        (
            relations_3wm,
            relations_4wm,
            coeffs_3wm,
            coeffs_4wm,
        ) = prepare_relations_coefficients(
            terms_3wm, terms_4wm, self.twpa.epsilon, self.twpa.xi
        )

        gammas = np.empty((n_freq, n_modes), dtype=np.complex128)
        for i in range(n_freq):
            kappas = np.array([mode_params[mode]["k"][i] for mode in mode_labels])
            alphas = np.array([mode_params[mode]["alpha"][i] for mode in mode_labels])
            gammas[i] = 1j * kappas - alphas

        # Broadcast initial conditions to (n_freq, n_modes)
        if y0.ndim == 1:
            y0_broadcast = np.repeat(y0, n_freq).reshape((-1, n_freq)).T.copy()
        else:
            y0_broadcast = np.ascontiguousarray(y0)

        I_tuples_array = no_reflections_cmes_solve(
            x,
            y0_broadcast,
            gammas,
            relations_3wm,
            relations_4wm,
            coeffs_3wm,
            coeffs_4wm,
        )

        I_tuples_array = I_tuples_array * np.exp(np.einsum("ij,k->ijk", gammas, x))

        # Calculate gain
        signal_idx = mode_labels.index(signal_mode)
        if y0.ndim == 1:
            initial_signal = y0[signal_idx]
        else:
            initial_signal = y0[:, signal_idx]

        gain_db = 10 * np.log10(
            np.abs(I_tuples_array[:, signal_idx, -1] / initial_signal) ** 2
        )

        # Create mode maps
        mode_map = {label: idx for idx, label in enumerate(mode_labels)}
        reverse_mode_map = {idx: label for label, idx in mode_map.items()}

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
                "reverse_mode_map": reverse_mode_map,
                "signal_mode": signal_mode,
                "pump_mode": pump_mode,
                "idler_mode": idler_mode,
            },
        }

    @analysis_function
    def quantum_efficiency(
        self,
        signal_freqs: FloatArray,
        pump: Optional[float] = None,
        Is0: float = 1e-6,
        Ip0: Optional[float] = None,
        mode_array_config: str = "basic_3wm",
        model: Union[str, GainModel] = GainModel.GENERAL_NO_REFLECTIONS,
        signal_mode: str = "s",
        pump_mode: str = "p",
        idler_mode: str = "i",
        initial_pump_amplitudes_fwd: Optional[Union[List[float], np.ndarray]] = None,
        initial_pump_amplitudes_bwd: Optional[Union[List[float], np.ndarray]] = None,
        thin: int = 100,
        **forward_backward_kwargs,
    ) -> dict[str, Any]:
        """
        Compute the quantum efficiency (QE) of the TWPA in the lossless approximation.

        Basis convention
        ----------------
        The CME propagates current amplitudes I_m. The correct photon-flux
        amplitude in this formulation is

            a_m = I_m / sqrt(k_m)

        because the conserved Manley-Rowe quantity verified numerically from
        the CME is  |I_s|^2/k_s - |I_k|^2/k_k = const  (not omega-weighted).
        This is consistent with circuit QED where power ~ |I|^2 * Z_0 and
        the group velocity v_g = domega/dk enters the photon-flux density.

        Converting the current-basis transfer amplitudes to the photon basis:
            G_s            = |I_s_out/I_s_in|^2          (k_s cancels, same mode)
            |v_k|^2        = (k_k/k_s) * |I_s_out/I_k_in|^2   (noise channel k)
            Bogoliubov:    G_s - (k_s/k_k) * |I_k_out_col/I_s_in|^2 = 1

        where |I_k_out_col|^2 is the output of mode k from the signal-seeded run.

        QE:
            eta        = G_s / (G_s + sum_k |v_k|^2)
            eta_ideal  = G_s / (2*G_s - 1)               (Caves 1982 quantum limit,
                         exact for 2-mode amplifier, from signal run only)
            qe_norm    = eta / eta_ideal                  (1.0 = quantum limited)
            eta_MR     = k_s / k_p                       (high-gain asymptote)

        Parameters
        ----------
        signal_freqs : array_like
        pump : float, optional
        Is0 : float, default 1e-6
        Ip0 : float, optional
        mode_array_config : str, default 'basic_3wm'
        signal_mode : str, default 's'
        pump_mode : str, default 'p'
        idler_mode : str, default 'i'
        thin : int, default 100

        Returns
        -------
        dict with keys: signal_freqs, pump_freq, gain_db, qe, qe_normalized,
        eta_ideal, eta_manley_rowe, added_noise, noise_channels,
        bogoliubov_residual, mode_info.
        """
        if Ip0 is not None:
            self.twpa.Ip0 = Ip0
        if pump is None:
            pump = self.data["optimal_pump_freq"]

        signal_freqs = np.asarray(signal_freqs)
        n_freq = len(signal_freqs)

        mode_array = self.get_mode_array(mode_array_config)
        mode_labels = list(mode_array.modes.keys())
        n_modes = len(mode_labels)
        mode_to_idx = {label: idx for idx, label in enumerate(mode_labels)}
        signal_idx = mode_to_idx[signal_mode]
        pump_idx = mode_to_idx[pump_mode]

        N_tot = self.twpa.N_tot
        x = np.linspace(0, N_tot, int(N_tot / thin), endpoint=True)

        mode_array.update_frequencies({pump_mode: pump})
        mode_params = mode_array.process_frequency_array(signal_mode, signal_freqs)

        # kappas_array[i, m] = k_m at signal-frequency index i, shape (n_freq, n_modes)
        # alphas_array[i, m] = alpha_m (amplitude loss rate) - used for loss correction
        kappas_array = np.array(
            [[mode_params[m]["k"][i] for m in mode_labels] for i in range(n_freq)]
        )
        alphas_array = np.array(
            [[mode_params[m]["alpha"][i] for m in mode_labels] for i in range(n_freq)]
        )
        reflections = np.array(
            [[mode_params[m]["gamma"][i] for m in mode_labels] for i in range(n_freq)]
        )
        gammas = -alphas_array + 1j * kappas_array

        terms_3wm = mode_array.get_rwa_terms(power=2)
        terms_4wm = mode_array.get_rwa_terms(power=3)
        (
            relations_3wm,
            relations_4wm,
            coeffs_3wm,
            coeffs_4wm,
        ) = prepare_relations_coefficients(
            terms_3wm, terms_4wm, self.twpa.epsilon, self.twpa.xi
        )

        def _run_cme(y0: np.ndarray) -> np.ndarray:
            """Return full physical amplitude trajectory, shape (n_freq, n_modes, n_x)."""
            if model == GainModel.GENERAL_NO_REFLECTIONS:
                I = no_reflections_cmes_solve(
                    x, y0, gammas, relations_3wm, relations_4wm, coeffs_3wm, coeffs_4wm
                )
            else:
                y0_bwd = np.full(y0[0].shape, 1e-14, dtype=np.complex128)
                I_tuples = general_cmes_solve_fb_cutoff(
                    x,
                    y0[0],
                    y0_bwd,
                    gammas,
                    reflections,
                    relations_3wm,
                    relations_4wm,
                    coeffs_3wm,
                    coeffs_4wm,
                    save_all_passes=False,
                    **forward_backward_kwargs,
                )
                I = I_tuples[:n_freq]
                I[:, :, -1] = I[:, :, -1] * (1 + reflections)

            return I * np.exp(np.einsum("ij,k->ijk", 1j * kappas_array, x))

        # ---- signal gain run - keep full trajectory ----
        y0_s = np.full((n_freq, n_modes), 1e-14, dtype=np.complex128)
        y0_s[:, pump_idx] = self.twpa.Ip0
        y0_s[:, signal_idx] = Is0

        # I_traj_s: shape (n_freq, n_modes, n_x)
        log.info("Running for mode %s", signal_mode)
        I_traj_s = _run_cme(y0_s)
        I_out_s = I_traj_s[:, :, -1]

        # current-basis squared transfers from signal seed, shape (n_freq, n_modes)
        T_s = np.abs(I_out_s / Is0) ** 2
        U_sq = T_s[:, signal_idx]  # G_s, shape (n_freq,)

        ks = kappas_array[:, signal_idx]  # k_s, shape (n_freq,)
        kp = kappas_array[:, pump_idx]  # k_p, shape (n_freq,)

        # Vacuum injected at position x into the signal mode is amplified by the
        # gain of the remaining section [x, L]:
        #   G_s(x -> L) = |I_s(L)|^2 / |I_s(x)|^2

        Is_traj_sq = np.abs(I_traj_s[:, signal_idx, :]) ** 2  # (n_freq, n_x)

        # alpha_s per frequency, shape (n_freq,)
        alpha_s = alphas_array[:, signal_idx]

        # The loss integrand requires the true physical amplitude |A_s(x)|^2.
        # _run_cme returns output_s(x) = A_s(x) * exp(+alpha_s*x), so:
        #   |A_s(x)|^2 = |output_s(x)|^2 * exp(-2*alpha_s*x)
        attenuation_s = np.exp(-2.0 * alpha_s[:, np.newaxis] * x[np.newaxis, :])
        As_traj_sq = (
            Is_traj_sq * attenuation_s
        )  # true physical |A_s(x)|^2, (n_freq, n_x)

        integrand = (
            2.0
            * alpha_s[:, np.newaxis]
            / np.where(
                As_traj_sq > 0, As_traj_sq, As_traj_sq.max(axis=1, keepdims=True)
            )
        )
        # N_add_loss: input-referred vacuum noise from distributed signal-path loss
        N_add_loss = U_sq * np.abs(Is0) ** 2 * np.trapz(integrand, x, axis=1)

        # ---- noise channels + per-channel distributed loss ----
        dependent_modes = mode_array.get_dependent_modes(signal_mode)
        noise_mode_labels = [
            m for m in dependent_modes if m != pump_mode and m != signal_mode
        ]
        signal_transmission = np.exp(-2 * alphas_array[:, signal_idx] * self.twpa.N_tot)
        noise_channels: dict[str, np.ndarray] = {}
        bogoliubov_per_mode: dict[str, np.ndarray] = {}
        gain_db = 10 * np.log10(U_sq * signal_transmission)
        gain_db_per_mode = {signal_mode: gain_db}
        total_noise = np.zeros(n_freq)
        N_add_loss_channels = np.zeros(n_freq)  # loss noise from all noise channels

        for noise_mode in noise_mode_labels:
            log.info("Running for mode %s", noise_mode)
            noise_idx = mode_to_idx[noise_mode]
            kk = kappas_array[:, noise_idx]
            kk_over_ks = np.where(ks != 0, kk / ks, 1.0)

            y0_k = np.full((n_freq, n_modes), 1e-14, dtype=np.complex128)
            y0_k[:, pump_idx] = self.twpa.Ip0
            y0_k[:, noise_idx] = Is0

            I_traj_k = _run_cme(y0_k)
            I_out_k = I_traj_k[:, :, -1]
            V_k_current_sq = np.abs(I_out_k[:, signal_idx] / Is0) ** 2

            # Photon-basis noise: (k_k/k_s) * |V_k_current|^2
            v_k_sq = kk_over_ks * V_k_current_sq
            noise_channels[noise_mode] = v_k_sq
            total_noise += v_k_sq

            # Per-mode Bogoliubov contribution (lossless, signed):
            sign = -np.sign(mode_array.symbolic_expressions[noise_mode][signal_mode])
            bogoliubov_per_mode[noise_mode] = sign * v_k_sq

            # Distributed loss noise from channel k, photon-basis, input-referred.
            # A noise photon lost from mode k at position x has probability of
            # reaching the signal output proportional to the remaining parametric
            # gain G_{s<-k}(x->L).
            #
            # At the moment, only the added loss noise from signal and idler is computed,
            # Assuming that the current trajectectory is the same for the two
            # This avoids the singularity that arises from using the signal output
            # trajectory from the noise-seeded run (which starts at zero at x=0),
            # and avoids the 1/|A_k|^2 amplification that occurs when mode k is
            # weakly excited (e.g. far-detuned sidebands 20 dB below the signal).
            if noise_mode == idler_mode:
                alpha_k = alphas_array[:, noise_idx]
                # attenuation_db = 20*np.log10(np.exp(-alpha_k*x[-1]))
                # att_threshold = -100 # for numerical stability
                # alpha_k = np.where(attenuation_db>att_threshold, alpha_k, -np.log(10**(att_threshold/20)))
                attenuation_k = np.exp(-2.0 * alpha_k[:, np.newaxis] * x[np.newaxis, :])
                Ik_traj_sq = np.abs(I_traj_k[:, noise_idx, :]) ** 2 * attenuation_k
                integrand_k = (
                    2.0
                    * alpha_k[:, np.newaxis]
                    / np.where(
                        Ik_traj_sq > 0,
                        Ik_traj_sq,
                        Ik_traj_sq.max(axis=1, keepdims=True),
                    )
                )
                N_add_k = v_k_sq * np.abs(Is0) ** 2 * np.trapz(integrand_k, x, axis=1)
                N_add_loss_channels += N_add_k
            gain_db_per_mode[noise_mode] = 10 * np.log10(v_k_sq * signal_transmission)
        # Total Bogoliubov residual: G_s - sum_k sign*(kk/ks)*|V_k_row|^2 - 1 = 0
        bogoliubov_residual = U_sq - sum(bogoliubov_per_mode.values()) - 1.0

        total_noise_with_loss = total_noise + N_add_loss + N_add_loss_channels

        qe = U_sq / (U_sq + total_noise_with_loss)
        added_noise = total_noise_with_loss / U_sq / 2
        noise_channels_out = noise_channels  # already in the same frame

        qe_ideal = U_sq / (2.0 * U_sq - 1.0)
        qe_normalized = np.where(qe_ideal > 0, qe / qe_ideal, 0.0)

        return {
            "signal_freqs": signal_freqs,
            "pump_freq": pump,
            "gain_db": gain_db,
            "qe": qe,
            "qe_normalized": qe_normalized,
            "qe_ideal": qe_ideal,
            "added_noise": added_noise,
            "noise_channels": noise_channels_out,
            "gain_db_per_mode": gain_db_per_mode,
            "bogoliubov_per_mode": bogoliubov_per_mode,
            "bogoliubov_residual": bogoliubov_residual,
            "loss_noise_signal": N_add_loss / U_sq / 2,
            "loss_noise_channels": N_add_loss_channels / U_sq / 2,
            "mode_info": {
                "mode_array": mode_array_config,
                "signal_mode": signal_mode,
                "pump_mode": pump_mode,
                "idler_mode": idler_mode,
                "noise_modes": noise_mode_labels,
            },
        }

    @analysis_function
    def bandwidth(
        self,
        gain_reduction: float = 3,
        **gain_kwargs,
    ) -> dict[str, Any]:
        """
        Compute information about the compound gain bandwidth.

        Args:
            gain_reduction (float): Difference from maximum gain used as threshold.
            **gain_kwargs: Additional arguments to pass to the gain function.

        Returns:
            dict: Bandwidth analysis results.
        """
        if gain_kwargs or "gain" not in self.data.keys():
            self.gain(**gain_kwargs)

        gain_data = self.data["gain"]
        pump_freq = gain_data["pump_freq"]
        gain_db = gain_data["gain_db"]
        signal_freqs = gain_data["signal_freqs"]

        max_g_idx = np.argmax(gain_db)
        max_g = gain_db[max_g_idx]
        ok_idx = np.where(gain_db >= max_g - gain_reduction)[0]
        ok_freqs = signal_freqs[ok_idx]
        df = signal_freqs[1] - signal_freqs[0]
        freq_diff = np.diff(ok_freqs)

        extremes_indices = np.where(freq_diff >= 2 * df)[0]
        all_roots = [ok_freqs[0]]
        for idx in extremes_indices:
            all_roots.extend([ok_freqs[idx], ok_freqs[idx + 1]])
        all_roots.append(ok_freqs[-1])

        return {
            "pump_freq": pump_freq,
            "bandwidth_edges": all_roots,
            "total_bw": len(ok_freqs) * df,
            "max_gain": max_g,
            "reduced_gain": max_g - gain_reduction,
            "mean_gain": np.mean(gain_db[ok_idx]),
            "bw_freqs": ok_freqs,
        }

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
        gain_data = self.data["gain"]

        return plot_mode_currents(gain_data, **kwargs)

    def plot_phase_matching(self, **kwargs) -> Axes:
        """Plot phase matching profile of twpa."""
        if "phase_matching" not in self.data:
            raise RuntimeError(
                "Phase matching data not found, please run analysis function."
            )
        phase_matching_data = self.data["phase_matching"]
        return plot_phase_matching(
            phase_matching_data["pump_freqs"],
            phase_matching_data["signal_freqs"],
            phase_matching_data["delta"],
            freqs_unit=self.unit,
            **kwargs,
        )

    def plot_quantum_efficiency(self, **kwargs) -> np.ndarray:
        if "quantum_efficiency" not in self.data:
            raise RuntimeError("Run quantum_efficiency() first.")
        return plot_quantum_efficiency(
            self.data["quantum_efficiency"],
            freqs_unit=self.unit,
            **kwargs,
        )
