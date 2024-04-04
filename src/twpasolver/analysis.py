"""Analysis classes."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
from CyRK import nbrk_ode
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeFloat,
    PrivateAttr,
    field_validator,
)
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

from twpasolver.file_utils import read_file, save_to_file
from twpasolver.logging import log
from twpasolver.mathutils import (
    CMEode_complete,
    CMEode_undepleted,
    compute_phase_matching,
)
from twpasolver.models import TWPA


def analysis_function(
    func,
):
    """Wrap functions for analysis."""

    def wrapper(self, *args, save=True, **kwargs):
        self.update_base_data()
        function_name = func.__name__
        if function_name == "parameter_sweep":
            function_name = args[0] + "_sweep"
        log.info(f"Running {function_name}")
        result = func(self, *args, **kwargs)
        if save:
            if function_name in self.data.keys():
                log.info(
                    f"Data for '{function_name}' output already present in analysis data, will be overwritten."
                )
            self.data[function_name] = result
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

    data_file: Optional[str] = None
    run: List[ExecutionRequest] = Field(default_factory=list)
    _allowed_functions: List[str] = PrivateAttr(default_factory=list)
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
        if self.data_file is None:
            data_file = "data"
        else:
            data_file = self.data_file
        save_to_file(data_file, self.data, writer=writer)

    def execute(self):
        """Run analysis."""
        for request in self.run:
            function_name = request.name
            if function_name not in self._allowed_functions:
                raise ValueError(
                    f"Function '{function_name}' is not supported, choose between {self._allowed_functions}."
                )
            function = getattr(self, function_name)
            _ = function(*request.args, **request.kwargs)

        if self.data_file:
            self.save_data()

    @analysis_function
    def parameter_sweep(
        self, function: str, target: str, values: List, *args, **kwargs
    ):
        """Run an analysis function multiple times for different values of a parameter."""
        results = {}
        for value in values:
            fn = getattr(self, function)
            kwargs.update({target: value})
            results[value] = fn(*args, save=False, **kwargs)
        return {target: results}


class TWPAnalysis(Analyzer):
    """Runner for standard analysis routines for twpa models."""

    model_config = ConfigDict(validate_assignment=True)
    twpa: TWPA
    freqs_arange: Tuple[NonNegativeFloat, NonNegativeFloat, NonNegativeFloat] = Field(
        frozen=True
    )
    freqs_unit: Literal["Hz", "kHz", "MHz", "GHz"] = "GHz"
    _previous_state: Dict[str, Any] = PrivateAttr()
    _allowed_functions = PrivateAttr(["phase_matching", "gain", "bandwidth"])
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
            self.data["abcd"] = cell.abcd
            self.data["S21"] = cell.get_s_par()[:, 1, 0]
            self.data["k"] = np.unwrap(np.angle(self.data["S21"])) / self.twpa.N_tot
            s21_db = np.log(np.abs(self.data["S21"]))
            s21_db_diff = s21_db[1:] - s21_db[:-1]
            self.data["stopband_freqs"] = (
                freqs[np.argmin(s21_db_diff)],
                freqs[np.argmax(s21_db_diff)],
            )
            self._previous_state = current_state

    @analysis_function
    def phase_matching(self) -> Dict[str, Any]:
        """Build phase matching graph and triplets."""
        freqs = self.data["freqs"]
        ks = self.data["k"]
        min_p_idx = np.where(freqs == self.data["stopband_freqs"][1])[0][0]
        max_p_idx = -1
        min_s_idx = np.where(freqs == self.data["stopband_freqs"][0])[0][0]
        signal_f = freqs[:min_s_idx]
        signal_k = ks[:min_s_idx]
        pump_f = freqs[min_p_idx:max_p_idx]
        pump_k = ks[min_p_idx:max_p_idx]
        deltas, f_triplets, k_triplets = compute_phase_matching(
            signal_f, pump_f, signal_k, pump_k, self.twpa.chi
        )
        optimal_pump_idx = np.argmin(f_triplets[1:, 1] - f_triplets[:-1, 1])
        return {
            "delta": deltas,
            "triplets": {"f": f_triplets, "k": k_triplets},
            "pump_freqs": pump_f,
            "signal_freqs": signal_f,
            "optimal_pump_freq": f_triplets[optimal_pump_idx, 0],
        }

    @analysis_function
    def gain(
        self,
        signal_arange: Tuple[float, float, float],
        pump: Optional[float] = None,
        model: Literal["complete", "undepleted"] = "complete",
        Is0: float = 1e-6,
        scipy: bool = False,
    ):
        """Compute expected gain."""
        if pump is None:
            if "phase_matching" not in self.data.keys():
                self.phase_matching()
            pump = self.data["phase_matching"]["optimal_pump_freq"]
        freqs = self.data["freqs"]
        ks = self.data["k"]
        signal_freqs = np.arange(*signal_arange)
        idler_freqs = pump - signal_freqs
        pump_k = np.interp(pump, freqs, ks)
        signal_k = np.interp(signal_freqs, freqs, ks)
        idler_k = np.interp(idler_freqs, freqs, ks)
        N_tot = self.twpa.N_tot
        x_span = (0, N_tot)
        x = np.linspace(0, N_tot, N_tot + 1)
        y0 = np.array([self.twpa.Ip0, Is0, 0], dtype=np.complex128)
        model_func = {"complete": CMEode_complete, "undepleted": CMEode_undepleted}.get(
            model, "complete"
        )
        gains = []
        Iss = []
        Iis = []
        Ips = []
        xs = 0
        for i, _ in enumerate(signal_freqs):
            if scipy:
                sol = solve_ivp(
                    model_func,
                    x_span,
                    y0,
                    args=(
                        pump_k,
                        signal_k[i],
                        idler_k[i],
                        self.twpa.xi,
                        self.twpa.epsilon,
                    ),
                    t_eval=x,
                    method="RK45",
                    first_step=1,
                    max_step=1000,
                )
                xs, Ip, Is, Ii = sol.t, sol.y[0, :], sol.y[1, :], sol.y[2, :]
            else:
                xs, y_sol, _, _ = nbrk_ode(
                    model_func,
                    x_span,
                    y0,
                    args=(
                        pump_k,
                        signal_k[i],
                        idler_k[i],
                        self.twpa.xi,
                        self.twpa.epsilon,
                    ),
                    atol=1e-16,
                    max_num_steps=1000,
                    first_step=1,
                    t_eval=x,
                    rk_method=1,
                )
                Ip, Is, Ii = y_sol[0, :], y_sol[1, :], y_sol[2, :]
            gains.append(np.abs(Is / complex(Is0)) ** 2)
            Ips.append(Ip)
            Iss.append(Is)
            Iis.append(Ii)
        gains_arr = np.asarray(gains)
        gains_db = 10 * np.log10(gains_arr)
        return {
            "pump_freq": pump,
            "signal_freqs": signal_freqs,
            "x": np.asarray(xs),
            "Ip": np.array(Ips),
            "Ii": np.array(Iss),
            "Ii": np.array(Iis),
            "gain": gains_arr,
            "gain_db": gains_db,
        }

    @analysis_function
    def bandwidth(
        self,
        signal_arange: Tuple[float, float, float],
        gain_reduction: float = 3,
        **gain_kwargs,
    ):
        """Compute frequency bandwidth."""
        if gain_kwargs or "gain" not in self.data.keys():
            self.gain(signal_arange, **gain_kwargs)
        pump_freq = self.data["gain"]["pump_freq"]
        gains_db = self.data["gain"]["gain_db"][:, -1]
        signal_freqs = self.data["gain"]["signal_freqs"]
        max_g_idx = np.argmax(gains_db)
        max_g = gains_db[max_g_idx]
        interp_fn = lambda x: np.abs(
            np.interp(x, signal_freqs, gains_db) - max_g + gain_reduction
        )
        triplets = self.data["phase_matching"]["triplets"]["f"]
        matched_triplet = triplets[np.searchsorted(triplets[:, 0], pump_freq)]

        root_left = minimize(
            interp_fn,
            matched_triplet[1],
            method="Powell",
            bounds=[(0, matched_triplet[1])],
        )["x"][0]

        root_right = minimize(
            interp_fn,
            matched_triplet[2],
            method="Powell",
            bounds=[(matched_triplet[2], pump_freq)],
        )["x"][0]

        return {
            "pump_freq": pump_freq,
            "signal_freqs": (root_left, root_right),
            "bw": root_left + root_right,
            "max_gain": max_g,
            "reduced_gain": max_g - gain_reduction,
            "mean_gain": np.mean(
                gains_db[(signal_freqs >= root_left) & (signal_freqs <= root_right)]
            ),
        }
