"""Analysis classes."""

from abc import ABC, abstractmethod
from functools import partial
from time import strftime
from typing import Any, Dict, List, Literal, Optional

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, field_validator

from twpasolver.file_utils import read_file, save_to_file
from twpasolver.logging import log
from twpasolver.mathutils import cme_solve, compute_phase_matching
from twpasolver.models import TWPA
from twpasolver.plotting import plot_gain, plot_phase_matching, plot_response
from twpasolver.typing import FrequencyArange


def analysis_function(
    func,
):
    """Wrap functions for analysis."""

    def wrapper(self, *args, save=True, save_name: Optional[str] = None, **kwargs):
        self.update_base_data()
        function_name = func.__name__
        log.info(f"Running {function_name}")
        result = func(self, *args, **kwargs)
        if save:
            if function_name in self.data.keys():
                log.info(
                    f"Data for '{function_name}' output already present in analysis data, will be overwritten."
                )
            if save_name is None:
                save_name = function_name
                kwargs.update({"save_name": save_name})
            self.data[save_name] = result
        return result

    return wrapper


class ExecutionRequest(BaseModel):
    """General stucture of function execution request."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    name: str
    args: List[Any] = Field(default_factory=list)
    kwargs: Dict[str, Any] = Field(default_factory=dict)


class Analyzer(BaseModel, ABC):
    """Base class for structured analysis."""

    data_file: str = Field(default_factory=partial(strftime, "data_%m_%d_%H_%M_%S"))
    run: List[ExecutionRequest] = Field(default_factory=list)
    data: Dict[str, Any] = Field(default_factory=dict, exclude=True)

    def model_post_init(self, __context: Any) -> None:
        """Run analysis if list of ExecutionRequest is not empty."""
        if self.run:
            self.execute()

    @abstractmethod
    def update_base_data(self):
        """Check and update base data of the class if necessary."""

    @classmethod
    def from_file(cls, filename: str):
        """Load analysis from file."""
        analysis_dict = read_file(filename, writer="json")
        return cls(**analysis_dict)

    def dump_to_file(self, filename: str):
        """Dump analysis to file."""
        analysis_dict = self.model_dump()
        save_to_file(filename, analysis_dict, writer="json")

    def save_data(self, writer="hdf5"):
        """Save data to file."""
        save_to_file(self.data_file, self.data, writer=writer)

    def execute(self):
        """Run analysis."""
        for request in self.run:
            function = getattr(self, request.name)
            _ = function(*request.args, **request.kwargs)

    @analysis_function
    def parameter_sweep(
        self, function: str, target: str, values: List, *args, **kwargs
    ):
        """Run an analysis function multiple times for different values of a parameter."""
        results = {target: []}
        kwargs.update({"save": False})
        fn = getattr(self, function)
        for i, value in enumerate(values):
            kwargs.update({target: value})
            fn_res = fn(*args, **kwargs)
            if i == 0:
                results.update({key: [item] for key, item in fn_res.items()})
            else:
                for key, item in fn_res.items():
                    results[key].append(item)
            results[target].append(value)
        return results


class TWPAnalysis(Analyzer):
    """Runner for standard analysis routines for twpa models."""

    model_config = ConfigDict(validate_assignment=True)
    twpa: TWPA
    freqs_arange: FrequencyArange
    freqs_unit: Literal["Hz", "kHz", "MHz", "GHz"] = "GHz"
    _previous_state: Dict[str, Any] = PrivateAttr()
    _unit_multipliers = PrivateAttr({"Hz": 1, "kHz": 1e3, "MHz": 1e6, "GHz": 1e9})

    @field_validator("twpa", mode="before", check_fields=True)
    @classmethod
    def load_model_from_file(cls, twpa: str | TWPA) -> TWPA:
        """Try loading twpa model from file."""
        if isinstance(twpa, str):
            try:
                twpa = TWPA.from_file(twpa)
            except:
                raise ValueError("Input string mut be valid path to model file.")
        return twpa  # type: ignore[return-value]

    def update_base_data(self):
        """Update data from twpa model."""
        current_state = self.model_dump()
        if (
            not hasattr(self, "_previous_state")
            or current_state != self._previous_state
        ):
            log.info("Computing base parameters.")
            unit_multiplier = self._unit_multipliers[self.freqs_unit]
            freqs = np.arange(*self.freqs_arange)
            cell = self.twpa.get_cell(freqs * unit_multiplier)
            self.data["freqs"] = freqs
            self.data["abcd"] = np.asarray(cell.abcd)
            self.data["S21"] = cell.get_s_par()[:, 1, 0]
            s21_db = 20 * np.log(np.abs(self.data["S21"]))
            self.data["S21_db"] = s21_db
            s21_db_diff = s21_db[1:] - s21_db[:-1]
            stopband_start_idx = np.argmin(s21_db_diff)
            self.data["stopband_freqs"] = [
                freqs[stopband_start_idx],
                freqs[np.argmax(s21_db_diff)],
            ]

            # Fit low frequency phase response to correctly unrwap at f=0 Hz
            if self.freqs_arange[0] > freqs[stopband_start_idx] / 10:
                log.warn(
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

    def _estimate_optimal_pump(self):
        """Estimate optimal pump frequency."""
        freqs = self.data["freqs"]
        ks = self.data["k"]
        min_p_idx = np.where(freqs == self.data["stopband_freqs"][1])[0][0]
        pump_f = freqs[min_p_idx:-1]
        pump_k = ks[min_p_idx:-1]
        signal_f = pump_f / 2
        signal_k = np.interp(signal_f, freqs, ks)
        deltas = np.abs(pump_k - 2 * signal_k + self.twpa.chi * (pump_k - 4 * signal_k))
        return pump_f[np.argmin(deltas)]

    @analysis_function
    def phase_matching(self, thin: int = 20) -> Dict[str, Any]:
        """Build phase matching graph and triplets."""
        freqs = self.data["freqs"]
        ks = self.data["k"]
        min_p_idx = np.where(freqs == self.data["stopband_freqs"][1])[0][0]
        max_p_idx = -1
        min_s_idx = np.where(freqs == self.data["stopband_freqs"][0])[0][0]
        signal_f = freqs[:min_s_idx:thin]
        signal_k = ks[:min_s_idx:thin]
        pump_f = freqs[min_p_idx:max_p_idx:thin]
        pump_k = ks[min_p_idx:max_p_idx:thin]
        deltas, f_triplets, k_triplets = compute_phase_matching(
            signal_f, pump_f, signal_k, pump_k, self.twpa.chi
        )
        return {
            "delta": deltas,
            "triplets": {"f": f_triplets, "k": k_triplets},
            "pump_freqs": pump_f,
            "signal_freqs": signal_f,
        }

    @analysis_function
    def gain(
        self,
        signal_arange: FrequencyArange,
        pump: Optional[float] = None,
        Is0: float = 1e-6,
        Ip0: Optional[float] = None,
        thin: int = 100,
    ):
        """Compute expected gain as a function of position in the TWPA."""
        N_tot = self.twpa.N_tot
        if thin > N_tot:
            thin = N_tot
        if Ip0 is not None:
            self.twpa.Ip0 = Ip0
        if pump is None:
            pump = self.data["optimal_pump_freq"]
        freqs = self.data["freqs"]
        ks = self.data["k"]
        signal_freqs = np.arange(*signal_arange)
        idler_freqs = pump - signal_freqs
        pump_k = np.interp(pump, freqs, ks)
        signal_k = np.interp(signal_freqs, freqs, ks)
        idler_k = np.interp(idler_freqs, freqs, ks)
        x = np.linspace(0, N_tot, int(N_tot / thin), endpoint=True)
        y0 = np.array([self.twpa.Ip0, Is0, 0], dtype=np.complex128)

        I_triplets = cme_solve(
            signal_k, idler_k, x, y0, pump_k, self.twpa.xi, self.twpa.epsilon
        )
        gain_db = 10 * np.log10(np.abs(I_triplets[:, 1, -1] / y0[1]) ** 2)

        return {
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
    ):
        """Compute frequency bandwidth."""
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
            "ok_idx": ok_idx,
        }

    def plot_response(self, **kwargs):
        """Plot response of twpa."""
        if "k_star" not in self.data:
            self.update_base_data()
        return plot_response(
            self.data["freqs"],
            self.data["S21_db"],
            self.data["k_star"],
            freqs_unit=self.freqs_unit,
            **kwargs,
        )

    def plot_gain(self, **kwargs):
        """Plot gain of twpa."""
        if "gain" not in self.data:
            raise RuntimeError("Gain data not found, please run analysis function.")
        gain_data = self.data["gain"]
        return plot_gain(
            gain_data["signal_freqs"],
            gain_data["gain_db"],
            freqs_unit=self.freqs_unit,
            **kwargs,
        )

    def plot_phase_matching(self, **kwargs):
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
            freqs_unit=self.freqs_unit,
            **kwargs,
        )
