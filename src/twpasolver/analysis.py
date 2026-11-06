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
    bloch_mode_tables,
    edge_basis_tables,
    edge_reference_frequency,
    in_stopband,
    lossless_copy,
    process_weights,
    reactive_angle_deg,
    stopband_edges,
)
from twpasolver.cmes.bloch_fb import bloch_fb_segmented_cmes_solve, segmented_phase
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

    SIMPLE : "simple"
        Traditional coupled mode equations of a homogeneous line, forward waves
        only. The supercell enters only through the Bloch wavenumber; the
        couplings are those of a uniform medium (scaled by k), without the
        intracell current profiles, so they are not exact for loaded lines,
        especially near stopbands. Ignores reflections and the port/connector
        models. Fast; useful for quick sweeps and comparisons with the
        literature. Segmented devices are interpolated linearly between segment
        midpoints (a smooth taper).

    BLOCH : "bloch"
        Forward-backward model in the Bloch-mode basis: exact Bloch couplings
        with the intracell current profiles, counter-propagating waves and
        reflections, exact boundary conditions with the port/connector models,
        fixed edge basis for modes close to a stopband edge. Segmented devices
        (whole supercells per segment) are solved on per-segment Bloch modes
        joined by exact junction conditions. Configured by ``FBSolverConfig``.

    The name ``"general_no_reflections"`` of earlier versions is accepted for
    ``"simple"`` with a deprecation warning.
    """

    SIMPLE = "simple"
    BLOCH = "bloch"

    @classmethod
    def _missing_(cls, value):
        if value == "general_no_reflections":
            log.warning(
                "Gain model 'general_no_reflections' is now called 'simple'; "
                "the old name will be removed in a future version."
            )
            return cls.SIMPLE
        return None


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


def _harmonic_limits(
    split: np.ndarray,
    W: np.ndarray,
    kd_pi: np.ndarray,
    base: int,
    e_edge: float,
) -> np.ndarray:
    """
    Per-mode harmonic limits of ``FBSolverConfig.m_max='auto'``.

    ``base`` for every mode, raised to the Bragg order ``m = round(k d / pi)``
    of a mode inside a stopband (``|arg W| > 45 deg``) or close to its edge
    (``split < e_edge``, twice ``FBSolverConfig.edge_threshold``): its counter-propagating component, at
    ``-m pi / d``, is reached only through ``G_m``. The zone centre
    (``k d / pi < 0.5``) is excluded. All arrays share the shape
    ``(..., n_modes)``.
    """
    zone_centre = kd_pi < 0.5
    special = ((split < e_edge) | (reactive_angle_deg(W) > 45.0)) & ~zone_centre
    order = np.round(kd_pi).astype(np.int64)
    return np.where(special, np.maximum(base, order), base).astype(np.int64)


def _stopband_guard_weights(
    split: np.ndarray,
    W: np.ndarray,
    kd_pi: np.ndarray,
    e_edge: float,
) -> np.ndarray:
    """
    Per-mode weights of ``FBSolverConfig.simulate_stopband_modes=False``.

    1 for ``split >= e_edge``, smoothstep down to 0 at the edge on the passband
    side, and 0 inside a stopband (``|arg W| > 45 deg``). The zone centre
    (``k d / pi < 0.5``), where the splitting also vanishes but the Bloch basis
    stays valid, keeps weight 1. All arrays share the shape ``(..., n_modes)``.
    """
    zone_centre = kd_pi < 0.5
    t = np.clip(np.asarray(split, dtype=np.float64) / e_edge, 0.0, 1.0)
    w = t * t * (3.0 - 2.0 * t)
    w = np.where(reactive_angle_deg(W) > 45.0, 0.0, w)
    return np.where(zone_centre, 1.0, w)


def _edge_basis_entries(
    device: Any,
    split: np.ndarray,
    kappa: np.ndarray,
    freqs_hz: np.ndarray,
    n_cells: int,
    injected: np.ndarray,
    e_edge: float,
    in_gap: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, Optional[dict[str, Any]]]:
    """
    Modes of one uniform device (or segment) to expand on the fixed edge basis.

    Parameters
    ----------
    device : TWPA
        Uniform device or segment; its supercell defines the Bloch modes.
    split, kappa, freqs_hz : (n_freq, n_modes)
        Bloch splitting ``|sinh(gamma d)|``, kernel wavenumber (``Im`` of the
        kernel gamma, per unit cell) and mode frequency (Hz).
    n_cells : int
        Unit cells per supercell.
    injected : (n_modes,) bool
        Modes with an input, never flagged.
    e_edge : float
        Threshold on the splitting, also the anchor of the fixed basis.
    in_gap : (n_freq, n_modes) bool, optional
        Entries inside a stopband to flag as well, whatever their splitting
        (``FBSolverConfig.simulate_stopband_modes``).

    Returns
    -------
    near_injected : (n_modes,) bool
        Injected modes that lie below the threshold somewhere.
    flag : (n_freq, n_modes) bool
        Entries expanded on the fixed basis.
    entries : dict or None
        ``ii, nn`` (indices of the flagged entries), ``f_mode``, ``f_ref``,
        ``tab`` (:func:`edge_basis_tables`) and ``missing`` (mode indices for
        which no stopband edge was found, kept in the native basis); None when
        nothing is flagged (``ii`` is empty when no flagged entry has an edge)
        and ``outside`` (mode indices with entries below the threshold that lie
        neither in a stopband nor in the switch-over window of an edge, e.g.
        at a zone-folding point where the bands touch without a gap; kept in
        the native basis).

    An entry below the threshold is expanded on the fixed basis only if it lies
    inside a stopband or between the nearest edge and its reference frequency,
    the window where the splitting of that edge stays below ``e_edge``. The
    splitting alone also vanishes where two bands touch without opening a gap
    (zone folding of a supercell made of repeated periods); the Bloch modes are
    distinct there and the native basis is exact, while a fixed basis anchored
    at a distant edge would carry the wrong carrier wavenumber.
    """
    # The splitting also vanishes at the zone centre (k -> 0, low frequency),
    # where T -> I and the two Bloch modes stay distinct (W finite): no
    # stopband edge there, the native basis is exact.
    zone_centre = np.abs(kappa) * n_cells / np.pi < 0.5
    near = (split < e_edge) & ~zone_centre
    if in_gap is not None:
        near |= in_gap & ~zone_centre
    near_injected = injected & near.any(axis=0)
    flag = near & ~injected[None, :]
    if not flag.any():
        return near_injected, flag, None
    twpa_ll = lossless_copy(device)
    ii, nn = np.nonzero(flag)
    f_mode = freqs_hz[ii, nn]
    # Edge index: k d = m pi at the edge, m with the parity of the edge.
    kd_pi = np.abs(kappa[ii, nn]) * n_cells / np.pi
    m_edge = np.zeros(len(f_mode), dtype=np.int64)
    f_ref = np.empty(len(f_mode))
    f_edge = np.full(len(f_mode), np.nan)
    found = np.ones(len(f_mode), dtype=bool)
    # Every edge near the flagged frequencies, found in one scan; each entry
    # takes the nearest one. An edge farther than the scanned span may not be
    # the nearest, so those entries are rescanned with a wider span, and the
    # per-frequency search is the last resort.
    par_q = np.zeros(len(f_mode), dtype=np.int64)
    todo = np.ones(len(f_mode), dtype=bool)
    gap_entry = in_stopband(twpa_ll, f_mode)
    outside = np.zeros(len(f_mode), dtype=bool)
    for span in (1.0e9, 4.0e9, 8.0e9):
        if not todo.any():
            break
        e_tab, ref_tab, par_tab = stopband_edges(twpa_ll, f_mode[todo], e_edge, span)
        if not len(e_tab):
            continue
        idx = np.flatnonzero(todo)
        near = np.argmin(np.abs(e_tab[None, :] - f_mode[idx, None]), axis=1)
        ok = (np.abs(e_tab[near] - f_mode[idx]) <= span) & np.isfinite(ref_tab[near])
        f_ref[idx[ok]] = ref_tab[near[ok]]
        f_edge[idx[ok]] = e_tab[near[ok]]
        par_q[idx[ok]] = par_tab[near[ok]]
        todo[idx[ok]] = False
    for q, f in enumerate(f_mode):
        if not todo[q]:
            parity = int(par_q[q])
        else:
            try:
                f_ref[q], f_edge[q], parity = edge_reference_frequency(
                    twpa_ll, f, e_edge
                )
            except ValueError:
                # No stopband edge around f (e.g. a closed gap): no
                # exceptional point, the native basis is kept.
                found[q] = False
                continue
        # Inside a stopband, or inside the switch-over window [f_edge, f_ref]
        # of the edge; otherwise the small splitting is a band touching
        # without a gap and the native basis is kept.
        lo, hi = sorted((f_edge[q], f_ref[q]))
        if not (gap_entry[q] or lo <= f <= hi):
            outside[q] = True
            continue
        odd = 0 if parity > 0 else 1
        m_edge[q] = 2 * int(np.round((kd_pi[q] - odd) / 2)) + odd
        if abs(kd_pi[q] - m_edge[q]) > 0.5:
            log.warning(
                "Edge basis: a mode at %.4f GHz has k d / pi = %.3f, far from its "
                "edge index %d.",
                f / 1e9,
                kd_pi[q],
                m_edge[q],
            )
    missing = np.unique(nn[~found])
    outside_modes = np.unique(nn[outside])
    found &= ~outside
    if not found.all():
        flag[ii[~found], nn[~found]] = False
        ii, nn, f_mode = ii[found], nn[found], f_mode[found]
        f_ref, m_edge = f_ref[found], m_edge[found]
    tab = (
        edge_basis_tables(device, twpa_ll, f_mode, f_ref, m_edge)
        if found.any()
        else None
    )
    entries = {
        "ii": ii,
        "nn": nn,
        "f_mode": f_mode,
        "f_ref": f_ref,
        "tab": tab,
        "missing": missing,
        "outside": outside_modes,
    }
    return near_injected, flag, entries


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

    def _bloch_segments(self) -> tuple[list, np.ndarray]:
        """
        Segments and segment boundaries (unit cells) seen by the Bloch FB model.

        A uniform device is a single segment spanning the whole line.
        """
        if self._is_segmented():
            segments = list(self.twpa.cells)
            bounds = np.asarray(self.data["segments"]["boundaries"], dtype=float)
        else:
            segments = [self.twpa]
            bounds = np.array([0.0, float(self.twpa.N_tot)])
        return segments, bounds

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

    @staticmethod
    def _log_edge_basis_skipped(labels, near_injected, entries, e_edge):
        """Warnings of the edge basis: injected modes near an edge, edges not found."""
        if near_injected.any():
            log.warning(
                "Edge basis: mode(s) %s with an input lie close to a stopband edge "
                "and keep their native Bloch basis; their description there is "
                "unreliable.",
                [labels[j] for j in np.flatnonzero(near_injected)],
            )
        if entries is not None and len(entries["missing"]):
            log.warning(
                "Edge basis: no stopband edge found near mode(s) %s with "
                "|sinh(gamma d)| < %g at some points; they keep their native "
                "Bloch basis there.",
                [labels[j] for j in entries["missing"]],
                e_edge,
            )
        if entries is not None and len(entries.get("outside", ())):
            log.info(
                "Edge basis: mode(s) %s have |sinh(gamma d)| < %g at some points "
                "outside any stopband and edge window (bands touching without a "
                "gap); they keep their native Bloch basis there.",
                [labels[j] for j in entries["outside"]],
                e_edge,
            )

    def _fb_stopbands(
        self,
        bloch: dict[str, Any],
        cme_inputs: dict[str, Any],
        gammas: np.ndarray,
        injected: np.ndarray,
        fb_config: FBSolverConfig,
    ) -> dict[str, Any]:
        """
        Stopband treatment and harmonic limits of the Bloch forward-backward models.

        ``fb_config.edge_threshold`` marks the modes close to an edge;
        ``fb_config.simulate_stopband_modes`` either expands them, and every
        mode inside a stopband, on the fixed edge basis
        (:meth:`_apply_edge_basis_segmented`), or switches their processes off
        with :func:`_stopband_guard_weights`. Modes with an input are never
        changed. ``m_max='auto'`` gives per-mode harmonic limits
        (:func:`_harmonic_limits`, window ``2 * edge_threshold``).

        Returns a dict with the kernel ``gammas``, ``edge`` (edge-basis tables or
        None), ``m_max`` (int or per-mode table) and the process weights ``pw3``,
        ``pw4`` (None when every process has weight 1).
        """
        e = fb_config.edge_threshold
        labels = cme_inputs["mode_labels"]
        kd_pi = (
            np.abs(np.transpose(gammas.imag, (0, 2, 1)))
            * np.asarray(bloch["n_cells"])[None, :, None]
            / np.pi
        )
        if fb_config.m_max == "auto":
            m_max = _harmonic_limits(
                bloch["split"],
                bloch["W"],
                kd_pi,
                solver_config.BLOCH_M_MAX_AUTO_BASE,
                2.0 * (e if e is not None else 0.10),
            )
            bloch["m_lim"] = m_max
        else:
            m_max = fb_config.m_max
        out: dict[str, Any] = dict(
            gammas=gammas, edge=None, m_max=m_max, pw3=None, pw4=None
        )
        if e is None:
            return out
        if fb_config.simulate_stopband_modes:
            edge = self._apply_edge_basis_segmented(
                bloch, cme_inputs, injected, e, True
            )
            if edge is not None:
                out["edge"] = edge
                out["gammas"] = edge["gammas"]  # kernel and carrier: -1j kappa_e
            return out
        weights = _stopband_guard_weights(bloch["split"], bloch["W"], kd_pi, e)
        lead = tuple(range(weights.ndim - 1))
        near_inj = injected & (weights < 1.0).any(axis=lead)
        if near_inj.any():
            log.warning(
                "Stopband guard: mode(s) %s with an input lie in a stopband or close "
                "to its edge and are kept in the native Bloch basis; their "
                "description there is unreliable.",
                [labels[j] for j in np.flatnonzero(near_inj)],
            )
        weights[..., injected] = 1.0
        bloch["edge_weights"] = weights
        if np.all(weights == 1.0):
            return out
        out["pw3"] = process_weights(cme_inputs["relations_3wm"], weights)
        out["pw4"] = process_weights(cme_inputs["relations_4wm"], weights)
        return out

    def _apply_edge_basis_segmented(
        self,
        bloch: dict[str, Any],
        cme_inputs: dict[str, Any],
        injected: np.ndarray,
        threshold: Optional[float],
        gap: bool = False,
    ) -> Optional[dict[str, Any]]:
        """
        Fixed ("edge") Bloch basis for modes close to a stopband edge.

        Every segment (a uniform device is one segment) is treated with its own
        supercell: a mode without input whose splitting in segment ``s`` is
        below ``threshold``, or that lies inside a stopband (``gap``), is
        expanded there on the lossless Bloch pair of that segment's supercell,
        anchored on the passband side of the nearest edge
        (:func:`_edge_basis_entries`). Modes with an input are never flagged.
        The junction coefficients follow from the overwritten ``Zb_plus`` and
        ``Zb_minus`` and stay exact. A mode flagged in at least one segment
        (``edge_mode``) is solved exactly along the whole line; in its native
        segments the generator is ``diag(-alpha, alpha)``.

        Returns None when nothing is flagged, otherwise the kernel gammas
        (``(n_freq, n_modes, n_seg)``), ``edge_K`` (``(n_freq, n_seg, n_modes,
        2, 2)``, per unit cell), ``edge_flag`` (``(n_freq, n_seg, n_modes)``) and
        ``edge_mode`` (``(n_freq, n_modes)``); diagnostics are stored in
        ``bloch``.
        """
        if threshold is None:
            return None
        e_edge = float(threshold)
        gammas = cme_inputs["gammas"]  # (n_freq, n_modes, n_seg)
        labels = cme_inputs["mode_labels"]
        segments, _ = self._bloch_segments()
        n_freq, n_seg, n_modes = bloch["split"].shape
        freqs_hz = bloch["freqs_hz"]
        flag = np.zeros((n_freq, n_seg, n_modes), dtype=bool)
        near_inj = np.zeros(n_modes, dtype=bool)
        missing: set[int] = set()
        outside: set[int] = set()
        gammas_k = gammas.copy()
        edge_K = np.zeros((n_freq, n_seg, n_modes, 2, 2), dtype=np.complex128)
        f_ref_tab = np.full((n_freq, n_seg, n_modes), np.nan)
        ham = np.zeros((n_freq, n_seg, n_modes))
        cache: dict[Any, Any] = {}
        for s, seg in enumerate(segments):
            split_s = bloch["split"][:, s]
            kappa_s = gammas[:, :, s].imag
            in_gap = (reactive_angle_deg(bloch["W"][:, s]) > 45.0) if gap else None
            key = (
                repr([c.model_dump() for c in seg.cells]),
                np.ascontiguousarray(kappa_s).tobytes(),
                np.ascontiguousarray(split_s).tobytes(),
            )
            if key not in cache:
                cache[key] = _edge_basis_entries(
                    seg,
                    split_s,
                    kappa_s,
                    freqs_hz,
                    int(bloch["n_cells"][s]),
                    injected,
                    e_edge,
                    in_gap=in_gap,
                )
            ni, fl, ent = cache[key]
            near_inj |= ni
            flag[:, s] = fl
            if ent is None:
                continue
            missing.update(int(j) for j in ent["missing"])
            outside.update(int(j) for j in ent["outside"])
            if len(ent["ii"]) == 0:
                continue
            ii, nn, tab = ent["ii"], ent["nn"], ent["tab"]
            n_nl = tab["phi_plus"].shape[-1]
            bloch["Zb_plus"][ii, s, nn] = tab["Zb_plus"]
            bloch["Zb_minus"][ii, s, nn] = tab["Zb_minus"]
            bloch["W"][ii, s, nn] = tab["W"]
            bloch["phi_plus"][ii, s, nn, :n_nl] = tab["phi_plus"]
            bloch["phi_minus"][ii, s, nn, :n_nl] = tab["phi_minus"]
            gammas_k[ii, nn, s] = -1j * tab["kappa_e"]
            edge_K[ii, s, nn] = tab["K"]
            f_ref_tab[ii, s, nn] = ent["f_ref"]
            ham[ii, s, nn] = tab["ham_defect"]
        self._log_edge_basis_skipped(
            labels,
            near_inj,
            {
                "missing": np.array(sorted(missing), int),
                "outside": np.array(sorted(outside), int),
            },
            e_edge,
        )
        if not flag.any():
            return None
        edge_mode = flag.any(axis=1)
        # Native segments of a mode solved exactly: decoupled directions with
        # the kernel attenuation (forward decays along +x, backward along -x).
        for i, n in zip(*np.nonzero(edge_mode)):
            for s in np.flatnonzero(~flag[i, :, n]):
                a = gammas_k[i, n, s].real
                edge_K[i, s, n] = np.array([[a, 0.0], [0.0, -a]], dtype=np.complex128)
        omegas = 2 * np.pi * freqs_hz
        bloch["prefac"] = -2j * omegas[:, None, :] / bloch["W"]
        seg_len = np.diff(bloch["seg_bounds"])
        bragg = np.abs(edge_K[..., 0, 1]) * seg_len[None, :, None]
        bloch.update(
            edge_flag=flag,
            edge_mode=edge_mode,
            edge_f_ref=f_ref_tab,
            edge_bragg_strength=bragg,
            edge_ham_defect=ham,
        )
        for n in np.flatnonzero(edge_mode.any(axis=0)):
            log.info(
                "Edge basis: mode %s at %d points, flagged in %d of %d segments, "
                "max Bragg strength |K01| L_seg = %.1f.",
                labels[n],
                int(edge_mode[:, n].sum()),
                int(flag[:, :, n].any(axis=0).sum()),
                n_seg,
                float(bragg[:, :, n].max()),
            )
        return {
            "gammas": gammas_k,
            "edge_K": edge_K,
            "edge_flag": flag,
            "edge_mode": edge_mode,
        }

    def _prepare_bloch_segmented(
        self,
        mode_array: ModeArray,
        cme_inputs: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Per-segment Bloch tables for the forward-backward solver.

        A uniform device is a single segment (``cme_inputs["gammas"]`` must then
        carry a segment axis of length 1). Every segment is a uniform ``TWPA``
        made of whole supercells; its Bloch
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
        segments, seg_bounds = self._bloch_segments()
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
            "seg_bounds": seg_bounds,
            "eig_residual": eig_res,
            "gamma_mismatch": g_mis,
            "base_3wm": np.stack([unit_3wm * seg.epsilon for seg in segments]),
            "base_4wm": np.stack([unit_4wm * seg.xi for seg in segments]),
            "prefac": -2j * omegas[:, None, :] / W,
            "freqs_hz": freqs_hz,
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
        model: Union[str, GainModel] = GainModel.SIMPLE,
        mode_array_config: str = "basic_3wm",
        signal_mode: str = "s",
        pump_mode: str = "p",
        idler_mode: str = "i",
        initial_amplitudes_fwd: Optional[Union[List[float], np.ndarray]] = None,
        initial_amplitudes_bwd: Optional[Union[List[float], np.ndarray]] = None,
        thin: int = 100,
        fb_config: Optional[FBSolverConfig] = None,
    ) -> dict[str, Any]:
        """
        Compute the TWPA gain using coupled mode equation models.

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
        model : GainModel or str, default 'simple'
            CME model (see :class:`GainModel`):
            * 'simple': traditional CMEs of a homogeneous line, forward waves
              only; fast, approximate couplings, no reflections or connectors.
            * 'bloch': forward-backward model in the Bloch-mode basis with exact
              couplings and boundary conditions; segmented devices are solved per
              segment with exact junction conditions.
        mode_array_config : str, default 'basic_3wm'
            Name of mode array configuration to use.
        signal_mode, pump_mode, idler_mode : str
            Mode labels within the mode array.
        initial_amplitudes_fwd, initial_amplitudes_bwd : array_like, optional
            Amplitudes of the waves injected at x=0 (traveling along +x) and at
            x=L (traveling along -x); override Is0 and Ip0. Backward inputs
            require the 'bloch' model; 'simple' raises a ValueError if any is
            nonzero.
        thin : int, default 100
            Position array thinning factor.
        fb_config : FBSolverConfig, optional
            Settings of the 'bloch' solver (default ``FBSolverConfig()``);
            ignored by 'simple'.

        Returns
        -------
        dict
            Gain analysis results with model-specific fields. The 'bloch'
            model also returns the
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

        if Ip0 is not None:
            self.twpa.Ip0 = Ip0
        if pump is None:
            pump = self.data["optimal_pump_freq"]

        signal_freqs = np.asarray(signal_freqs)
        N_tot = self.twpa.N_tot
        x = np.linspace(0, N_tot, int(N_tot / thin), endpoint=True, dtype=np.float64)

        if model == GainModel.BLOCH:
            return self._compute_bloch_gain(
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
    ) -> dict[str, Any]:
        """Gain with the traditional forward-only CMEs (``model="simple"``)."""
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
            "model": GainModel.SIMPLE.value,
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

    def _compute_bloch_gain(
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
        impedances Zb+ (forward) and Zb- (backward), so the boundary
        coefficients are constant. Segmented devices (a ``TWPA`` whose cells are all ``TWPA`` segments)
        are solved on per-segment Bloch modes joined by exact junction
        conditions; uniform devices are the one-segment case. See
        :mod:`twpasolver.cmes.bloch_fb`.
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
        if not segmented:
            # A uniform device is the one-segment case.
            cme_inputs = {**cme_inputs, "gammas": cme_inputs["gammas"][..., None]}
        bloch = self._prepare_bloch_segmented(mode_array, cme_inputs)

        Zout_in, Zout_out, abcd_in, abcd_out = self._build_port_tables(
            mode_params, mode_labels, n_freq
        )

        y0_fwd_broadcast = self._broadcast_ic(y0_fwd, n_freq)
        y0_bwd_broadcast = self._broadcast_ic(y0_bwd, n_freq)
        injected = np.any(np.abs(y0_fwd_broadcast) > 0, axis=0) | np.any(
            np.abs(y0_bwd_broadcast) > 0, axis=0
        )
        sb = self._fb_stopbands(
            bloch, cme_inputs, cme_inputs["gammas"], injected, fb_config
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
            passes=fb_config.passes,
            kappa_cutoff=fb_config.kappa_cutoff,
            m_max=sb["m_max"],
            process_weights_3wm=sb["pw3"],
            process_weights_4wm=sb["pw4"],
            convergence_threshold=fb_config.convergence_threshold,
            save_all_passes=fb_config.save_all_passes,
            anderson_depth=fb_config.anderson_depth,
            Ze=self.port_Z_ext,
            Zout_in=Zout_in,
            Zout_out=Zout_out,
            abcd_in=abcd_in,
            abcd_out=abcd_out,
            edge_K=None if edge is None else edge["edge_K"],
            edge_mode=None if edge is None else edge["edge_mode"],
            diagnostics=diag,
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
