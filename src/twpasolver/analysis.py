"""Analysis classes."""

import logging
from abc import ABC
from functools import cached_property
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeFloat,
    PrivateAttr,
    field_validator,
    model_validator,
)
from skrf import Network

from twpasolver.file_utils import read_file, save_to_file
from twpasolver.mathutils import compute_phase_matching
from twpasolver.models import TWPA
from twpasolver.typing import float_array


def analysis_function(func):
    """Wrap functions for analysis."""

    def wrapper(self, *args, **kwargs):
        function_name = func.__name__
        if function_name in self.data.keys():
            logging.warning(
                "Data for {function_name} output already present in analysis data, will be overwritten."
            )
        result = func(self, *args, **kwargs)
        self.data[function_name] = result
        return result

    return wrapper


class ExecutionRequest(BaseModel):
    """General stucture of function execution request."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    name: str
    parameters: Optional[Dict[str, Any]]


class Analyzer(BaseModel, ABC):
    """Base class for structured analysis."""

    data_file: Optional[str] = None
    run: List[ExecutionRequest] = Field(default_factory=list)
    _allowed_functions: List[str] = PrivateAttr(default_factory=list)
    data: Dict[str, Any] = Field(default_factory=dict, exclude=True)

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
            _ = function(**request.parameters)

        if self.data_file:
            self.save_data()


class TWPAnalysis(Analyzer):
    """Runner for standard analysis routines for twpa models."""

    # model_config = ConfigDict(ignored_types=[Network])
    twpa: TWPA = Field(frozen=True)
    freqs_arange: Tuple[NonNegativeFloat, NonNegativeFloat, NonNegativeFloat] = Field(
        frozen=True
    )
    _allowed_functions = PrivateAttr(["phase_matching", "gain", "bandwidth"])

    @field_validator("twpa", mode="before")
    @classmethod
    def load_model_from_file(cls, twpa: str | TWPA) -> TWPA:
        """Try loading twpa model from file."""
        if isinstance(twpa, str):
            try:
                twpa = TWPA.from_file(twpa)
            except:
                raise ValueError("Input string mut be valid path to model file.")
        return twpa

    @model_validator(mode="after")
    def update_base_data(self):
        """Update data from twpa model."""
        freqs = np.arange(*self.freqs_arange)
        cell = self.twpa.get_cell(freqs)
        self.data.update(**cell.as_dict())
        self.data["S21"] = cell.get_s_par()[:, 1, 0]
        self.data["k"] = np.unwrap(np.angle(self.data["S21"])) / self.twpa.N_tot
        return self

    @cached_property
    def network(self) -> Network:
        """Network object of model."""
        return self.twpa.get_network(np.arange(*self.freqs_arange))

    @analysis_function
    def phase_matching(self, min_pump_f, max_pump_f) -> Dict[str, float_array]:
        """Build phase matching graph and triplets."""
        freqs = self.data["freqs"]
        ks = self.data["k"]
        min_p_idx = np.searchsorted(freqs, min_pump_f)
        max_p_idx = np.searchsorted(freqs, max_pump_f)
        signal_f = freqs[:min_p_idx]
        signal_k = ks[:min_p_idx]
        pump_f = freqs[min_p_idx:max_p_idx]
        pump_k = ks[min_p_idx:max_p_idx]
        print(min_p_idx, max_p_idx)
        deltas, f_triplets, k_triplets = compute_phase_matching(
            signal_f, pump_f, signal_k, pump_k, self.twpa.chi
        )
        return {"delta": deltas, "f_trip": f_triplets, "k_trip": k_triplets}

    def gain(self):
        """Compute expected gain."""
        pass
