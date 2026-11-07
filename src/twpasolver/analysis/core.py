"""``TWPAnalysis``: the interface for simulating a TWPA."""

from typing import Any, List, Literal, Mapping, Optional, Tuple, Union

import numpy as np
from matplotlib.axes import Axes
from pydantic import ConfigDict, Field, PrivateAttr, TypeAdapter, field_validator

from twpasolver.analysis.gain import GainModel, solve_bloch, solve_simple
from twpasolver.analysis.linear import (
    LinearData,
    ParameterInterpolator,
    interpolator,
    linear_data,
    segments_of,
)
from twpasolver.analysis.phase_matching import phase_matching
from twpasolver.analysis.ports import Ports
from twpasolver.analysis.problem import build_problem
from twpasolver.analysis.results import GainResult, PhaseMatchingResult
from twpasolver.basemodel import BaseModel
from twpasolver.bonus_types import FloatArray, FrequencyArange, FrequencyList
from twpasolver.cmes.prep import CMEProblem
from twpasolver.cmes.solver_config import BlochSolverConfig, RKSolverConfig
from twpasolver.logger import log
from twpasolver.models import TWPA
from twpasolver.modes_rwa import ModeSet, as_mode_set
from twpasolver.plotting import plot_response
from twpasolver.twoport import TwoPortCell

_UNIT_MULTIPLIERS = {"Hz": 1, "kHz": 1e3, "MHz": 1e6, "GHz": 1e9, "THz": 1e12}

Grid = Union[Tuple[float, float, float], List[float], np.ndarray]


def _grid(spec: Grid) -> np.ndarray:
    """A tuple ``(start, stop, step)`` through ``numpy.arange``, else the explicit values."""
    if isinstance(spec, tuple):
        return np.arange(*spec)
    return np.asarray(spec, dtype=float)


class TWPAnalysis(BaseModel):
    """
    Interface for simulating a TWPA with coupled mode equations (CMEs).

    Frequencies are given and reported in ``unit``; the solvers work in Hz. The
    linear response is computed on demand (:attr:`linear`) and recomputed when
    ``twpa``, ``freqs`` or ``unit`` change; analysis methods return their results
    without storing them.
    """

    model_config = ConfigDict(validate_assignment=True, extra="forbid")
    twpa: TWPA
    freqs: Union[FrequencyArange, FrequencyList] = Field(
        description=(
            "Frequency grid of the linear response, in ``unit``: a tuple "
            "``(start, stop, step)`` passed to ``numpy.arange``, or a list or array "
            "of frequencies. Stored as the explicit list."
        )
    )
    unit: Literal["Hz", "kHz", "MHz", "GHz", "THz"] = "GHz"
    ports: Ports = Field(default_factory=Ports)

    _linear: Optional[LinearData] = PrivateAttr(default=None)
    _linear_key: Any = PrivateAttr(default=None)
    _interp: Optional[ParameterInterpolator] = PrivateAttr(default=None)

    @field_validator("twpa", mode="before")
    @classmethod
    def _load_model_from_file(cls, twpa: str | TWPA) -> TWPA:
        """Load the twpa model from a JSON file when given a path."""
        if isinstance(twpa, str):
            try:
                return TWPA.from_file(twpa)
            except Exception as exc:
                raise ValueError(
                    "Input string must be valid path to model file."
                ) from exc
        return twpa

    @field_validator("freqs", mode="before")
    @classmethod
    def _expand_grid(cls, v):
        """Expand an arange tuple and accept arrays; reject an empty grid."""
        if isinstance(v, tuple):
            start, stop, step = TypeAdapter(FrequencyArange).validate_python(v)
            v = np.arange(start, stop, step, dtype=np.float64)
        if isinstance(v, np.ndarray):
            v = v.tolist()
        if isinstance(v, list) and not v:
            raise ValueError("The frequency grid is empty.")
        return v

    @property
    def unit_multiplier(self) -> float:
        """Multiplier converting ``unit`` to Hz."""
        return _UNIT_MULTIPLIERS[self.unit]

    @property
    def f(self) -> np.ndarray:
        """Frequency grid in Hz."""
        return self.unit_multiplier * np.array(self.freqs, dtype=np.float64)

    @property
    def linear(self) -> LinearData:
        """Linear response on the grid, recomputed when ``twpa``, ``freqs`` or ``unit`` change."""
        key = (self.twpa.model_dump(), self.freqs, self.unit)
        if self._linear is None or key != self._linear_key:
            log.info("Computing base parameters.")
            self._linear = linear_data(self.twpa, self.f, self.unit_multiplier)
            self._interp = interpolator(self._linear)
            self._linear_key = key
        return self._linear

    def response(self, connectors: bool = True) -> TwoPortCell:
        """
        Linear two-port response of the device on the grid (``.s.S21``, ``.abcd``, ...).

        With ``connectors`` the connectors of ``ports`` are cascaded at both ends (the
        output one reversed: its port 2 faces the device) and the parameters refer to
        ``ports.Z_ext``, as the ``bloch`` gains do; otherwise the bare device, referred
        to ``twpa.Z0_ref`` as in :attr:`linear`.
        """
        T = np.asarray(self.linear.abcd)
        if not connectors:
            return TwoPortCell(self.f, abcd=T, Z0=self.twpa.Z0_ref)
        p = self.ports
        out = p.output if p.output is not None else p.input
        if p.input is not None:
            T = np.asarray(p.input.get_abcd(self.f)) @ T
        if out is not None:
            m = np.asarray(out.get_abcd(self.f))
            det = m[:, 0, 0] * m[:, 1, 1] - m[:, 0, 1] * m[:, 1, 0]
            T = T @ (m[:, ::-1, ::-1].transpose(0, 2, 1) / det[:, None, None])
        return TwoPortCell(self.f, abcd=T, Z0=p.Z_ext)

    def problem(
        self, modes: Union[str, ModeSet], tones: Mapping[str, Any]
    ) -> CMEProblem:
        """
        CME inputs of ``modes`` at the frequencies set by the independent ``tones``.

        Args:
            modes: A ``ModeSet`` or ``"basic_3wm"`` / ``"basic_4wm"``.
            tones: Frequency of every independent mode, in ``unit`` (scalars or
                arrays of one length).
        """
        lin = self.linear
        assert self._interp is not None
        return build_problem(
            as_mode_set(modes),
            tones,
            lin,
            self._interp,
            segments_of(self.twpa),
            self.unit_multiplier,
        )

    def phase_matching(
        self,
        signal_freqs: Grid,
        pump_freqs: Grid,
        process: Literal["PA", "FCU", "FCD"] = "PA",
        mixing: Literal["3WM", "4WM"] = "3WM",
        modes: Union[str, ModeSet] = "basic_3wm",
        signal_mode: str = "s",
        pump_mode: str = "p",
        idler_mode: str = "i",
        tones: Optional[Mapping[str, float]] = None,
        pump_Ip0: Optional[float] = None,
        relative: bool = True,
    ) -> PhaseMatchingResult:
        """
        Phase mismatch of a 3WM or 4WM process over pump and signal frequencies.

        Args:
            signal_freqs, pump_freqs: A tuple ``(start, stop, step)`` passed to
                ``numpy.arange``, or the explicit frequencies.
            process: ``"PA"`` (amplification), ``"FCU"`` or ``"FCD"`` (up/down
                conversion); the idler frequency is the one ``modes`` gives
                ``idler_mode``.
            mixing: ``"3WM"`` or ``"4WM"``.
            tones: Frequencies of further independent modes of ``modes``.
            pump_Ip0: Pump current of the Kerr term; None uses ``twpa.Ip0``, 0
                gives the linear mismatch.
            relative: Divide the mismatch by the signal wavenumber.

        ``optimal_pump_freq`` of the result is the pump with the smallest mean
        ``|delta|`` over the signals, so restrict them to the band of interest;
        pumps inside a stopband get NaN and are never chosen. Segmented devices
        give the length-weighted mean of the per-segment mismatch.
        """
        signal_f, pump_f = _grid(signal_freqs), _grid(pump_freqs)
        lin = self.linear
        return phase_matching(
            lambda p: self.problem(
                modes, {**(tones or {}), pump_mode: p, signal_mode: signal_f}
            ),
            signal_f,
            pump_f,
            lin.stopbands,
            lin.seg_N_tot,
            lin.seg_xi,
            self.twpa.Ip0 if pump_Ip0 is None else pump_Ip0,
            process,
            mixing,
            signal_mode,
            pump_mode,
            idler_mode,
            relative,
            self.unit,
        )

    def gain(
        self,
        signal_freqs: FloatArray,
        pump: float,
        Is0: float = 1e-7,
        Ip0: Optional[float] = None,
        model: Union[str, GainModel] = GainModel.SIMPLE,
        modes: Union[str, ModeSet] = "basic_3wm",
        signal_mode: str = "s",
        pump_mode: str = "p",
        tones: Optional[Mapping[str, float]] = None,
        initial_amplitudes_fwd: Optional[Union[List[float], np.ndarray]] = None,
        initial_amplitudes_bwd: Optional[Union[List[float], np.ndarray]] = None,
        thin: int = 100,
        bloch_config: Optional[BlochSolverConfig] = None,
        rk_config: Optional[RKSolverConfig] = None,
    ) -> GainResult:
        """
        Gain of the TWPA from the coupled mode equations.

        Args:
            signal_freqs: Signal frequencies, in ``unit``.
            pump: Pump frequency. There is no automatic choice; for an estimate use
                ``phase_matching(...).optimal_pump_freq``.
            Is0: Signal current injected at x=0 (A).
            Ip0: Pump current injected at x=0 (A); defaults to ``twpa.Ip0``.
            model: ``"simple"`` (forward-only textbook CMEs, no reflections or
                connectors) or ``"bloch"`` (forward-backward CMEs in the Bloch
                basis with exact couplings and boundaries); see :class:`GainModel`.
            modes: A ``ModeSet`` or ``"basic_3wm"`` / ``"basic_4wm"``.
            signal_mode, pump_mode: Independent modes swept over ``signal_freqs``
                and set to ``pump``.
            tones: Frequencies of further independent modes (e.g. a second pump).
            initial_amplitudes_fwd, initial_amplitudes_bwd: Currents of every mode
                incident at x=0 and at x=L, shape (n_modes,) or (n_freq, n_modes);
                ``initial_amplitudes_fwd`` overrides ``Is0`` and ``Ip0``. Backward
                inputs need ``"bloch"``.
            thin: Spacing in unit cells of the position grid (at least two points).
            bloch_config: Settings of ``"bloch"`` (default ``BlochSolverConfig()``).
            rk_config: Settings of the RK45 integrator (default ``RKSolverConfig()``).
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
        if model == GainModel.SIMPLE and (
            self.ports.input is not None or self.ports.output is not None
        ):
            log.warning("Model 'simple' ignores the connectors in ports; use 'bloch'.")
        signal_freqs = np.asarray(signal_freqs, dtype=float)
        pr = self.problem(
            modes, {**(tones or {}), pump_mode: pump, signal_mode: signal_freqs}
        )
        n_freq, n_modes = pr.n_freq, pr.n_modes

        def amplitudes(given: Any, fill: dict[str, float]) -> np.ndarray:
            y0: np.ndarray
            if given is None:
                y0 = np.zeros(n_modes, dtype=np.complex128)
                for label, value in fill.items():
                    y0[pr.mode_labels.index(label)] = value
            else:
                y0 = np.asarray(given, dtype=np.complex128)
            return np.ascontiguousarray(np.broadcast_to(y0, (n_freq, n_modes)))

        y0_fwd = amplitudes(
            initial_amplitudes_fwd,
            {pump_mode: self.twpa.Ip0 if Ip0 is None else Ip0, signal_mode: Is0},
        )
        N_tot = self.twpa.N_tot
        x = np.linspace(
            0, N_tot, max(2, int(N_tot / thin)), endpoint=True, dtype=np.float64
        )
        rk = rk_config if rk_config is not None else RKSolverConfig()
        meta = dict(
            unit=self.unit,
            signal_freqs=signal_freqs,
            pump_freq=pump,
            signal_mode=signal_mode,
            pump_mode=pump_mode,
        )
        if model == GainModel.BLOCH:
            return solve_bloch(
                pr,
                y0_fwd,
                amplitudes(initial_amplitudes_bwd, {}),
                x,
                self.ports,
                bloch_config if bloch_config is not None else BlochSolverConfig(),
                rk,
                **meta,
            )
        return solve_simple(pr, y0_fwd, x, rk, **meta)

    def plot_response(self, **kwargs) -> Axes:
        """Plot ``|S21|`` and ``k_star`` of the linear response."""
        lin = self.linear
        return plot_response(
            lin.freqs, lin.S21_db, lin.k_star, freqs_unit=self.unit, **kwargs
        )
