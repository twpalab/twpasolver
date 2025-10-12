"""
Improved analysis.py with proper forward-backward integration and cleaner notation.

Key improvements:
1. Added GENERAL_FB model to GainModel enum
2. Unified gain computation logic to reduce duplication
3. Clearer notation: gamma_bloch vs reflection_coeff
4. Better parameter handling and validation
"""

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
from twpasolver.cmes import (
    cme_general_solve_freq_array,
    cme_solve,
    cme_solve_forward_backward
)
from twpasolver.cmes_fb import cme_solve_forward_backward_cut
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
from twpasolver.twoport import TwoPortCell
import matplotlib.pyplot as plt
import matplotlib as mpl

class GainModel(str, enum.Enum):
    """
    Available CME models for gain analysis.

    Args
    ----
    MINIMAL_3WM : str
        Basic pump-signal-idler model. Fastest computation but limited accuracy.
        Suitable for quick design estimates and parameter space exploration.

    GENERAL_IDEAL : str
        Extended mode system without losses or reflections. Moderate computation
        time with improved accuracy over basic model. Includes pump harmonics
        and frequency conversion terms.

    GENERAL_LOSS_ONLY : str
        Extended mode system with device losses but no reflections. Similar
        computation time to ideal case with better accuracy for lossy devices.

    GENERAL : str
        Full extended mode system with losses and reflections. Highest accuracy
        but slowest computation (~10x general_ideal model). Use for quantitative
        predictions matching experimental data.

    GENERAL_FB : str
        Forward-backward propagation model with iterative solving for reflections.
        Most comprehensive model including multiple passes for reflection convergence.
        Slower than GENERAL but handles strong reflections more accurately.
    """

    MINIMAL_3WM = "minimal_3wm"
    GENERAL_IDEAL = "general_ideal"
    GENERAL_LOSS_ONLY = "general_loss_only"
    GENERAL = "general"
    GENERAL_FB = "general_fb"


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

    relations_3wm = extract_indices(terms_3wm) if terms_3wm else []
    relations_4wm = extract_indices(terms_4wm) if terms_4wm else []

    coeffs_3wm = (
        calculate_coefficients(terms_3wm, epsilon / 4) if terms_3wm else np.array([])
    )
    coeffs_4wm = (
        calculate_coefficients(terms_4wm, xi / 4) if terms_4wm else np.array([])
    )

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

    use_block: bool = False
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
        current_state = self.model_dump()
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
        self._mode_arrays["basic_3wm"] = ModeArrayFactory.create_basic_3wm(self.data)

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
        initial_amplitudes: Optional[Union[List[float], np.ndarray]] = None,
    ) -> np.ndarray:
        """Set initial conditions for CME solving."""
        if initial_amplitudes is None:
            n_modes = len(mode_array.modes)
            y0 = np.zeros(n_modes, dtype=np.complex128)
            mode_labels = list(mode_array.modes.keys())
            signal_idx = mode_labels.index(signal_mode)
            pump_idx = mode_labels.index(pump_mode)
            y0[pump_idx] = self.twpa.Ip0
            y0[signal_idx] = Is0
        else:
            y0 = np.array(initial_amplitudes, dtype=np.complex128)
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
        signal_mode: str = "s",
        pump_mode: str = "p",
        idler_mode: str = "i",
        mode_array_config: str = "basic_3wm",
        pump_Ip0: Optional[float] = None,
        signal_arange: Optional[Tuple[float, float, float]] = None,
        pump_arange: Optional[Tuple[float, float, float]] = None,
        thin: int = 20,
        relative=True
    ) -> dict[str, Any]:
        """Analyze phase matching conditions for nonlinear processes using mode arrays."""
        # Get the mode array
        mode_array = self.get_mode_array(mode_array_config)
        freqs = self.data["freqs"]

        if pump_arange:
            pump_f = np.arange(*pump_arange)[::thin]
        else:
            min_p_idx = np.where(freqs == self.data["stopband_freqs"][1])[0][0]
            max_p_idx = -1
            pump_f = freqs[min_p_idx:max_p_idx:thin]

        if signal_arange:
            signal_f = np.arange(*signal_arange)[::thin]
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
                    pump_k
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
                    deltas[:,i] =  deltas[:,i]/signal_k_array

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
        model: Union[str, GainModel] = GainModel.MINIMAL_3WM,
        mode_array_config: str = "basic_3wm",
        signal_mode: str = "s",
        pump_mode: str = "p",
        idler_mode: str = "i",
        initial_amplitudes: Optional[Union[List[float], np.ndarray]] = None,
        initial_amplitudes_fwd: Optional[Union[List[float], np.ndarray]] = None,
        initial_amplitudes_bwd: Optional[Union[List[float], np.ndarray]] = None,
        thin: int = 100,
        passes: int = 3,
        cutoff = 0.1
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
        model : GainModel or str, default 'minimal_3wm'
            CME model complexity:
            * 'minimal_3wm': Basic pump-signal-idler only (fastest)
            * 'general_ideal': Extended modes, no losses/reflections
            * 'general_loss_only': Extended modes with losses
            * 'general': Full model with losses and reflections
            * 'general_fb': Forward-backward model with iterative solving
        mode_array_config : str, default 'basic_3wm'
            Name of mode array configuration to use
        signal_mode, pump_mode, idler_mode : str
            Mode labels within the mode array
        initial_amplitudes : array_like, optional
            Initial current amplitudes for unidirectional models
        initial_amplitudes_fwd, initial_amplitudes_bwd : array_like, optional
            Initial conditions for forward-backward model
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

        # Route to appropriate computation method
        if model == GainModel.MINIMAL_3WM:
            return self._compute_minimal_3wm_gain(
                signal_freqs, pump, Is0, x, initial_amplitudes
            )
        elif model == GainModel.GENERAL_FB:
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
                cutoff=cutoff
            )
        else:
            return self._compute_general_gain(
                signal_freqs,
                pump,
                mode_array_config,
                signal_mode,
                pump_mode,
                idler_mode,
                x,
                model,
                initial_amplitudes,
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
        cutoff = False
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
        relations_coefficients = prepare_relations_coefficients(
            terms_3wm, terms_4wm, self.twpa.epsilon, self.twpa.xi
        )

        # Build CME data arrays
        cme_data_array = []
        gammas = []

        for i in range(n_freq):
            kappas = np.array([mode_params[mode]["k"][i] for mode in mode_labels])
            alphas = np.array([mode_params[mode]["alpha"][i] for mode in mode_labels])
            reflections = np.array(
                [mode_params[mode]["gamma"][i] for mode in mode_labels]
            )
            cme_data_array.append([kappas, alphas, reflections])
            gammas.append(-alphas + 1j * kappas)

        gammas = np.array(gammas)
        cme_data_array = np.array(cme_data_array, dtype=np.complex128)

        # Solve forward-backward CMEs
        if cutoff:
            I_tuples = cme_solve_forward_backward_cut(
            x,
            y0_fwd,
            y0_bwd,
            cme_data_array,
            relations_coefficients,
            thin=1,
            passes=passes,
            cutoff=cutoff
        )
        else:
            I_tuples = cme_solve_forward_backward(
            x,
            y0_fwd,
            y0_bwd,
            cme_data_array,
            relations_coefficients,
            thin=1,
            passes=passes,
        )

        # Extract results for all passes
        # I_tuples now has shape (2 * n_freq * passes, n_modes, len(x))
        I_tuples_all_passes = {}

        for pass_idx in range(passes):
            # Forward results for this pass
            start_fwd = pass_idx * n_freq
            end_fwd = (pass_idx + 1) * n_freq
            I_tuples_fwd_pass = I_tuples[start_fwd:end_fwd]

            # Backward results for this pass
            start_bwd = n_freq * passes + pass_idx * n_freq
            end_bwd = n_freq * passes + (pass_idx + 1) * n_freq
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
        I_tuples_fwd = I_tuples_all_passes[passes - 1]["forward"]
        I_tuples_bwd = I_tuples_all_passes[passes - 1]["backward"]

        # Calculate transmission coefficients and output currents
        transmission_coeffs = 1 + cme_data_array[:, 2, :]
        signal_idx = mode_labels.index(signal_mode)

        I_out_fwd = I_tuples_fwd[:, signal_idx, -1] * transmission_coeffs[:, signal_idx]
        I_out_bwd = I_tuples_bwd[:, signal_idx, -1] * transmission_coeffs[:, signal_idx]

        # Calculate gains for final pass
        gains_fwd = to_dB(np.abs(I_out_fwd / np.repeat(y0_fwd[signal_idx], n_freq)))
        gains_bwd = to_dB(np.abs(I_out_bwd / np.repeat(y0_bwd[signal_idx], n_freq)))
        gains_fwd_11 = to_dB(np.abs(I_out_bwd / np.repeat(y0_fwd[signal_idx], n_freq)))
        gains_bwd_22 = to_dB(np.abs(I_out_fwd / np.repeat(y0_bwd[signal_idx], n_freq)))

        # Calculate gains for all passes
        gains_all_passes = []
        for pass_idx in range(passes):
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
            "model": GainModel.GENERAL_FB.value,
            "pump_freq": pump,
            "signal_freqs": signal_freqs,
            "x": x,
            "I_triplets": I_tuples_fwd,  # Final pass forward results for backward compatibility
            "I_tuples_fwd": I_tuples_fwd,  # Final pass forward results
            "I_tuples_bwd": I_tuples_bwd,  # Final pass backward results
            "I_tuples_all_passes": I_tuples_all_passes,  # All passes results
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

    def _compute_general_gain(
        self,
        signal_freqs: FloatArray,
        pump: float,
        mode_array_config: str,
        signal_mode: str,
        pump_mode: str,
        idler_mode: str,
        x: FloatArray,
        model: GainModel,
        initial_amplitudes: Optional[Union[List[float], np.ndarray]] = None,
        Is0: float = 1e-6,
    ) -> dict[str, Any]:
        """Compute gain using the general coupled mode equations models."""
        # Get the mode array
        mode_array = self.get_mode_array(mode_array_config)
        mode_labels = list(mode_array.modes.keys())

        # Setup initial conditions
        y0 = self._setup_initial_conditions(
            mode_array, signal_mode, pump_mode, Is0, initial_amplitudes
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
        relations_coefficients = prepare_relations_coefficients(
            terms_3wm, terms_4wm, self.twpa.epsilon, self.twpa.xi
        )

        # Build CME data arrays based on model type
        cme_data_array = []
        reflections = model == GainModel.GENERAL
        with_loss = model in [GainModel.GENERAL_LOSS_ONLY, GainModel.GENERAL]

        for i in range(n_freq):
            kappas = np.array([mode_params[mode]["k"][i] for mode in mode_labels])

            if with_loss:
                alphas = np.array(
                    [mode_params[mode]["alpha"][i] for mode in mode_labels]
                )

                if reflections:
                    gammas = np.array(
                        [mode_params[mode]["gamma"][i] for mode in mode_labels]
                    )
                    # Calculate reflection terms
                    gammas_tilde = gammas * np.exp(1j * kappas * N_tot)
                    # gammas_tilde[np.abs(gammas) > 0.99] = 0
                    ts_reflection_neg = gammas_tilde / (1 - gammas_tilde**2)
                    ts_reflection = 1 / (1 - gammas_tilde**2)

                    cme_data_array.append(
                        [kappas, alphas, ts_reflection, ts_reflection_neg, gammas]
                    )
                else:
                    cme_data_array.append([kappas, alphas])
            else:
                # Ideal case: only kappas needed
                cme_data_array.append([kappas])

        # Solve the CMEs
        I_tuples_array = cme_general_solve_freq_array(
            x,
            y0,
            cme_data_array,
            relations_coefficients,
            thin=1,
            reflections=reflections,
            with_loss=with_loss and not reflections,
        )

        # Calculate gain
        signal_idx = mode_labels.index(signal_mode)
        if y0.ndim == 1:
            initial_signal = y0[signal_idx]
        else:
            initial_signal = y0[:, signal_idx]

        if reflections:
            # Extract signal mode parameters for transmission calculation
            gammas_signal = mode_params[signal_mode]["gamma"]
            kappas_signal = mode_params[signal_mode]["k"]

            # Calculate transmission coefficient
            ts_signal = 1 / (
                1 - gammas_signal**2 * np.exp(2j * kappas_signal * N_tot)
            )

            gain_db = 10 * np.log10(
                np.abs(
                    (I_tuples_array[:, signal_idx, -1] / initial_signal) ** 2
                    * (1 - gammas_signal**2) ** 2
                    * np.abs(ts_signal) ** 2
                )
            )
        else:
            # Simple gain calculation for ideal or loss-only cases
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

    def _compute_minimal_3wm_gain(
        self,
        signal_freqs: FloatArray,
        pump: float,
        Is0: float,
        x: FloatArray,
        initial_amplitudes: Optional[Union[List[float], np.ndarray]] = None,
    ) -> dict[str, Any]:
        """Compute gain using the standard 3WM model."""
        freqs = self.data["freqs"]
        ks = self.data["k"]

        # Calculate frequencies and wavenumbers
        idler_freqs = pump - signal_freqs
        pump_k = np.interp(pump, freqs, ks)
        signal_k = np.interp(signal_freqs, freqs, ks)
        idler_k = np.interp(idler_freqs, freqs, ks)

        # Setup initial conditions
        if initial_amplitudes is None:
            y0 = np.array([self.twpa.Ip0, Is0, 1e-14], dtype=np.complex128)
        else:
            y0 = np.array(initial_amplitudes, dtype=np.complex128)

        # Solve coupled mode equations
        I_tuples = cme_solve(
            signal_k, idler_k, x, y0, pump_k, self.twpa.xi, self.twpa.epsilon
        )

        # Calculate gain
        if y0.ndim == 1:
            initial_signal = y0[1]  # Signal is at index 1
        else:
            initial_signal = y0[:, 1]

        gain_db = 10 * np.log10(np.abs(I_tuples[:, 1, -1] / initial_signal) ** 2)

        return {
            "model": GainModel.MINIMAL_3WM.value,
            "pump_freq": pump,
            "signal_freqs": signal_freqs,
            "x": x,
            "I_tuples": I_tuples,
            "gain_db": gain_db,
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
