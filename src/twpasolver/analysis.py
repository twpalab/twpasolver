"""Analysis classes."""
"""Enhanced analysis classes with integrated ModeArray support."""

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
from twpasolver.cmes import cme_general_solve, cme_solve
from twpasolver.file_utils import read_file, save_to_file
from twpasolver.frequency import Frequencies
from twpasolver.logger import log
from twpasolver.models import TWPA
from twpasolver.modes_rwa import ModeArray, ModeArrayFactory
from twpasolver.plotting import (
    plot_gain,
    plot_mode_currents,
    plot_phase_matching,
    plot_response,
)


class GainModel(str, enum.Enum):
    """Available gain computation models."""

    STANDARD_3WM = "standard_3wm"
    GENERAL = "general"
    GENERAL_NO_GAMMA = "general_no_gamma"


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
    tuple
        Contains:
        - relations_3wm : List of mode indices for 3-wave mixing
        - relations_4wm : List of mode indices for 4-wave mixing
        - coeffs_3wm : numpy array of scaled coefficients for 3-wave mixing
        - coeffs_4wm : numpy array of scaled coefficients for 4-wave mixing
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


def analysis_function(
    func,
):
    """
    Wrap functions for analysis.

    Automatically saves results of each analysis function in data dictionary of an Analyzer class.
    All analysis functions must return a dictionary to be compatible with this wrapper.
    """

    @wraps(func)  # Necessary for correct functioning of sphinx autodoc
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
        """
        Run an analysis function multiple times for different values of a parameter.

        Args:
            function (str): Name of the analysis function to be executed multiple times.
            target (str): Name of the argument of the analysis function changed in the sweep.
            values (list): list of values for the sweep.

        Returns:
            dict: A dictionary containing:
                - The list of sweeped values.
                - The return dictionary of the analysis function,
                  with all items organized in lists with an additional dimension.

        """
        results = {target: []}  # type: dict[str, list[Any]]
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
    """Runner for standard analysis routines of TWPA models with enhanced ModeArray support."""

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
                    "Input string mut be valid path to model file."
                ) from exc
        return twpa  # type: ignore[return-value]

    def update_base_data(self) -> None:
        """
        Update data extracted from the twpa model.

        This function is triggered every time an analysis function is executed after changing a parameter of the model.
        """
        current_state = self.model_dump()
        if (
            not hasattr(self, "_previous_state")
            or current_state != self._previous_state
        ):
            log.info("Computing base parameters.")
            cell = self.twpa.get_cell(self.f)
            freqs = self.f / self.unit_multiplier
            self.data["freqs"] = freqs
            self.data["abcd"] = np.asarray(cell.abcd)
            self.data["S21"] = cell.s.S21
            s21_db = 20 * np.log(np.abs(self.data["S21"]))
            self.data["S21_db"] = s21_db
            s21_db_diff = s21_db[1:] - s21_db[:-1]
            stopband_start_idx = np.argmin(s21_db_diff)
            self.data["stopband_freqs"] = [
                freqs[stopband_start_idx],
                freqs[np.argmax(s21_db_diff)],
            ]
            Z0 = self.twpa.Z0_ref
            ac, b, c, d = (
                self.data["abcd"][:, 0, 0],
                self.data["abcd"][:, 0, 1],
                self.data["abcd"][:, 1, 0],
                self.data["abcd"][:, 1, 1],
            )
            Zbp = np.abs(2 * b / (ac - d - np.sqrt((ac + d) ** 2 - 4)))
            gammas = np.abs(Z0 - Zbp) / (Z0 + Zbp)
            self.data["Zb"] = Zbp
            self.data["gammas"] = gammas
            # Fit low frequency phase response to correctly unrwap at f=0 Hz
            if self.f[0] / self.unit_multiplier > freqs[stopband_start_idx] / 10:
                log.warning(
                    "Starting frequency is too high, unwrap of phase response might be imprecise."
                )
            k_full = -np.unwrap(np.angle(self.data["S21"])) / self.twpa.N_tot
            k_pars = np.polyfit(
                freqs[: int(stopband_start_idx / 10)],
                k_full[: int(stopband_start_idx / 10)],
                2,
            )
            self.data["k"] = k_full - k_pars[-1]
            self.data["k_star"] = k_full - np.polyval(k_pars[1:], freqs)
            self.data["optimal_pump_freq"] = self._estimate_optimal_pump()
            self._previous_state = current_state

            # Now that we have the base data, initialize standard mode arrays
            self._initialize_standard_mode_array()

    def _initialize_standard_mode_array(self) -> None:
        """Initialize standard mode arrays using the computed base data."""
        freqs = self.data["freqs"]
        kappas = self.data["k"]
        gammas = self.data["gammas"]

        # Create basic 3WM mode array
        self._mode_arrays["basic_3wm"] = ModeArrayFactory.create_basic_3wm(
            freqs, kappas, gammas
        )

    def get_mode_array(self, config: str = "basic_3wm") -> ModeArray:
        """
        Get a mode array by configuration name.

        Args:
            config: Name of the mode array configuration

        Returns:
            ModeArray: The requested mode array configuration
        """
        if config not in self._mode_arrays:
            raise ValueError(f"Mode array configuration '{config}' not found")
        return self._mode_arrays[config]

    def add_mode_array(self, name: str, mode_array: ModeArray) -> None:
        """
        Add a custom mode array to the analyzer.

        Args:
            name: Name to assign to the mode array
            mode_array: The mode array to add
        """
        self._mode_arrays[name] = mode_array

    def create_custom_mode_array(
        self,
        name: str,
        mode_labels: List[str],
        mode_directions: List[int],
        relations: List[List[str]],
    ) -> ModeArray:
        """
        Create and add a custom mode array to the analyzer.

        Args:
            name: Name to assign to the mode array
            mode_labels: List of mode labels
            mode_directions: List of mode directions (1 for forward, -1 for backward)
            relations: List of relations between modes

        Returns:
            ModeArray: The created mode array
        """
        mode_array = ModeArrayFactory.create_custom(
            self.data["freqs"],
            self.data["k"],
            self.data["gammas"],
            mode_labels,
            mode_directions,
            relations,
        )
        self.add_mode_array(name, mode_array)
        return mode_array

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
            pump_k
            - 2 * signal_k
            + (1 + np.abs(self.data["gammas"][min_p_idx:-1]) ** 2)
            * self.twpa.chi
            * (pump_k - 4 * signal_k)
        )
        return pump_f[np.argmin(deltas)]

    @analysis_function
    def phase_matching(
        self,
        process: Literal["PA", "FCU", "FCD"] = "PA",
        signal_mode: str = "s",
        pump_mode: str = "p",
        idler_mode: str = "i",
        mode_array_config: str = "basic_3wm",
        pump_Ip0: Optional[float] = None,
        signal_arange: Optional[Tuple[float]] = None,
        pump_arange: Optional[Tuple[float]] = None,
        thin: int = 20,
    ) -> dict[str, Any]:
        """
        Build phase matching profile for specified modes.

        Args:
            signal_mode: Label of the signal mode
            pump_mode: Label of the pump mode
            idler_mode: Label of the idler mode
            mode_array_config: Name of the mode array configuration to use
            thin: The step size to thin out the frequency and wavenumber arrays

        Returns:
            dict: A dictionary containing:
                - "delta" (array): Phase matching condition values
                - "triplets" (dict): Triplets satisfying phase matching
                - "pump_freqs" (array): Pump frequencies considered
                - "signal_freqs" (array): Signal frequencies considered
                - "mode_info" (dict): Information about the modes used
        """
        # Get the mode array
        mode_array = self.get_mode_array(mode_array_config)

        freqs = self.data["freqs"]
        ks = self.data["k"]
        gammas = self.data["gammas"]

        if pump_arange:
            pump_f = np.arange(*pump_arange)[::thin]
        else:
            # Determine frequency ranges for signal and pump
            min_p_idx = np.where(freqs == self.data["stopband_freqs"][1])[0][0]
            max_p_idx = -1
            pump_f = freqs[min_p_idx:max_p_idx:thin]

        if signal_arange:
            signal_f = np.arange(*signal_arange)[::thin]
        else:
            max_s_idx = np.where(freqs == self.data["stopband_freqs"][0])[0][0]
            signal_f = freqs[:max_s_idx:thin]

        chi = pump_Ip0**2 * self.twpa * xi / 8 if pump_Ip0 else self.twpa.chi

        f_triplets = []
        k_triplets = []

        deltas = np.empty(shape=(len(signal_f), len(pump_f)))

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
            for j, s_freq in enumerate(signal_f):
                mode_array.update_frequencies({signal_mode: s_freq})
                f_triplets.append(
                    [p_freq, s_freq, mode_array.get_mode(idler_mode).frequency]
                )
                # print(f_triplets[-1])
                pump_k, signal_k, idler_k = (
                    mode_array.get_mode(m).k
                    for m in [pump_mode, signal_mode, idler_mode]
                )
                k_triplets.append([pump_k, signal_k, idler_k])
                gamma = mode_array.get_mode(pump_mode).gamma
                deltas[j, i] = (
                    pump_k
                    + signal_sign * signal_k
                    + idler_sign * idler_k
                    + chi
                    * (1 + gamma**2)
                    * (pump_k + 2 * signal_sign * signal_k + 2 * idler_sign * idler_k)
                )

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
        model: Union[str, GainModel] = GainModel.STANDARD_3WM,
        mode_array_config: str = "basic_3wm",
        signal_mode: str = "s",
        pump_mode: str = "p",
        idler_mode: str = "i",
        initial_amplitudes: Optional[List[float]] = None,
        thin: int = 100,
    ) -> dict[str, Any]:
        """
        Compute expected gain with selected gain model.

        Args:
            signal_freqs: Array of signal frequencies to consider
            pump: The pump frequency (if None, uses the optimal pump frequency from data)
            Is0: Initial signal current (in A) for standard 3WM model
            Ip0: Initial pump current (in A) (if None, uses the current TWPA's Ip0)
            model: Gain model to use (standard_3wm, general or general without reflections)
            mode_array_config: Name of the mode array configuration to use
            signal_mode: Label of the signal mode
            pump_mode: Label of the pump mode
            idler_mode: Label of the idler mode (if None, will be determined from mode array relations)
            initial_amplitudes: Initial amplitudes for all modes in general model (if None, defaults will be used)
            thin: The step size to thin out the position array

        Returns:
            dict: A dictionary containing gain calculation results
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
        if isinstance(signal_freqs, list):
            signal_freqs = np.asarray(signal_freqs)

        # Get total number of cells
        N_tot = self.twpa.N_tot

        # Get position array
        x = np.linspace(0, N_tot, int(N_tot / thin), endpoint=True)

        # Use appropriate model for gain calculation
        if model == GainModel.STANDARD_3WM:
            return self._compute_standard_3wm_gain(signal_freqs, pump, Is0, x)
        else:
            reflections = True
            if model == GainModel.GENERAL_NO_GAMMA:
                reflections = False
            return self._compute_general_gain(
                signal_freqs,
                pump,
                mode_array_config,
                signal_mode,
                pump_mode,
                idler_mode,
                initial_amplitudes,
                x,
                reflections=reflections,
            )

    def _compute_general_gain(
        self,
        signal_freqs: FloatArray,
        pump: float,
        mode_array_config: str,
        signal_mode: str,
        pump_mode: str,
        idler_mode: str,
        initial_amplitudes: Optional[List[float]],
        x: FloatArray,
        reflections=True,
    ) -> dict[str, Any]:
        """Compute gain using the general coupled mode equations model with simplified processing."""
        # Get the mode array
        mode_array = self.get_mode_array(mode_array_config)

        # Mode indices for reference
        mode_labels = list(mode_array.modes.keys())
        signal_idx = mode_labels.index(signal_mode)
        pump_idx = mode_labels.index(pump_mode)
        idler_idx = mode_labels.index(idler_mode)

        N_tot = self.twpa.N_tot

        # Set up default initial conditions if not provided
        if initial_amplitudes is None:
            n_modes = len(mode_array.modes)
            y0 = np.zeros(n_modes, dtype=np.complex128)
            y0[pump_idx] = self.twpa.Ip0  # Pump amplitude
            y0[signal_idx] = 1e-6  # Small signal amplitude
        else:
            y0 = np.array(initial_amplitudes, dtype=np.complex128)

        # Update the pump frequency
        mode_array.update_frequencies({pump_mode: pump})

        # Process the array of signal frequencies
        mode_params = mode_array.process_frequency_array(signal_mode, signal_freqs)

        # Get RWA terms for 3-wave and 4-wave mixing (using cached results)
        terms_3wm = mode_array.get_rwa_terms(power=2)
        terms_4wm = mode_array.get_rwa_terms(power=3)

        # Prepare relations and coefficients
        relations_coefficients = prepare_relations_coefficients(
            terms_3wm, terms_4wm, self.twpa.epsilon, self.twpa.xi
        )

        # Prepare data for CME solver
        cme_data_batches = []
        for i in range(len(signal_freqs)):
            # Extract the parameters for this signal frequency
            kappas = np.array([mode_params[mode]["k"][i] for mode in mode_labels])
            if reflections:
                gammas = np.array(
                    [mode_params[mode]["gamma"][i] for mode in mode_labels]
                )
                gammas_tilde = gammas * np.exp(1j * kappas * N_tot)
                ts_reflection = 1 / (1 - gammas**2 * np.exp(2j * kappas * N_tot))
                ts_reflection_neg = gammas_tilde * ts_reflection
                cme_data_batches.append([kappas, ts_reflection, ts_reflection_neg])
            else:
                cme_data_batches.append([kappas])

        # Solve the CMEs for all signal frequencies
        I_triplets = []
        for dat in cme_data_batches:
            I_triplets.append(
                cme_general_solve(
                    x, y0, dat, relations_coefficients, thin=1, reflections=reflections
                )
            )
        I_triplets = np.array(I_triplets)
        # Calculate gain
        if reflections:
            # Extract signal mode gamma and calculate ts for gain calculation
            gammas_signal = mode_params[signal_mode]["gamma"]
            kappas_signal = mode_params[signal_mode]["k"]
            ts_signal = 1 / (1 - gammas_signal * np.exp(2j * kappas_signal * N_tot))

            gain_db = 10 * np.log10(
                np.abs(I_triplets[:, signal_idx, -1] / y0[signal_idx]) ** 2
                * (1 - np.abs(gammas_signal) ** 2) ** 2
                * np.abs(ts_signal) ** 2
            )
        else:
            gain_db = 10 * np.log10(
                np.abs(I_triplets[:, signal_idx, -1] / y0[signal_idx]) ** 2
            )

        # Create mode maps for better result interpretation
        mode_map = {label: idx for idx, label in enumerate(mode_labels)}
        reverse_mode_map = {idx: label for label, idx in mode_map.items()}

        return {
            "model": GainModel.GENERAL.value,
            "pump_freq": pump,
            "signal_freqs": signal_freqs,
            "x": x,
            "I_triplets": I_triplets,
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

    def _compute_standard_3wm_gain(
        self, signal_freqs: FloatArray, pump: float, Is0: float, x: FloatArray
    ) -> dict[str, Any]:
        """Compute gain using the standard 3WM model."""
        freqs = self.data["freqs"]
        ks = self.data["k"]

        # Calculate frequencies and wavenumbers
        idler_freqs = pump - signal_freqs
        pump_k = np.interp(pump, freqs, ks)
        signal_k = np.interp(signal_freqs, freqs, ks)
        idler_k = np.interp(idler_freqs, freqs, ks)

        # Initial conditions
        y0 = np.array([self.twpa.Ip0, Is0, 0], dtype=np.complex128)

        # Solve coupled mode equations
        I_triplets = cme_solve(
            signal_k,
            idler_k,
            x,
            y0,
            pump_k,  # type: ignore[arg-type]
            self.twpa.xi,
            self.twpa.epsilon,
        )

        # Calculate gain in dB
        gain_db = 10 * np.log10(np.abs(I_triplets[:, 1, -1] / y0[1]) ** 2)

        return {
            "model": GainModel.STANDARD_3WM.value,
            "pump_freq": pump,
            "signal_freqs": signal_freqs,
            "x": x,
            "I_triplets": I_triplets,
            "gain_db": gain_db,
        }

    @analysis_function
    def bandwidth(
        self,
        gain_reduction: float = 3,
        **gain_kwargs,
    ) -> dict[str, Any]:
        """
        Compute informations about the compund gain bandwidth.

        Args:
            gain_reduction (float): Difference from maximum gain used as threshold to determine the bandwidth edges.
            **gain_kwargs: Additional arguments to pass to the gain function.

        Returns:
            dict: A dictionary containing:
                - "pump_freq" (float): The pump frequency used in the gain calculation.
                - "bandwidth_edges" (array): The frequency edges of each gain profile section above the threshold.
                - "total_bw" (float): The total bandwidth.
                - "max_gain" (float): The maximum gain observed.
                - "reduced_gain" (float): The minimum gain level above the threshold.
                - "mean_gain" (float): The mean gain across the bandwidth range.
                - "bw_freqs" (array): Frequencies where gain is above the threshold.
        """
        if gain_kwargs or "gain" not in self.data.keys():
            self.gain(**gain_kwargs)
        pump_freq = self.data["gain"]["pump_freq"]
        gain_db = self.data["gain"]["gain_db"]
        signal_freqs = self.data["gain"]["signal_freqs"]
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
        return plot_mode_currents(
            gain_data,
            **kwargs,
        )

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
