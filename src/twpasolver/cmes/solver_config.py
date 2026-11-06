"""Global integrator settings and the forward-backward solver configuration."""

from __future__ import annotations

from typing import Literal

from pydantic import ConfigDict, Field, field_validator

from twpasolver.basemodel import BaseModel

# Global solver configuration
SOLVER_ATOL = 1e-12
SOLVER_RTOL = 1e-8
SOLVER_MAX_STEPS = 0
SOLVER_FIRST_STEP = 10
# Cap on the accepted RK45 steps per unit cell of the integration interval
# (device or segment length): ceil(length * SOLVER_MAX_STEPS_PER_CELL) + 1.
# A run that needs more is stopped and its unreached points are NaN, instead of
# shrinking the step forever. <= 0 disables the cap (only SOLVER_MAX_STEPS
# applies). It is compiled into the Numba kernels: after changing it here,
# clear the __pycache__ of twpasolver/cmes so the cached kernels are rebuilt.
SOLVER_MAX_STEPS_PER_CELL = 1

# Harmonic limit of the modes away from stopbands with FBSolverConfig.m_max =
# 'auto'.
BLOCH_M_MAX_AUTO_BASE = 1

# Quantum model (twpasolver.quantum) only: reactive-angle guard. Each mode gets
# a weight that is 1 for |arg W| below the first angle (degrees), 0 above the
# second, with a smoothstep in between: inside a gap W turns reactive and a
# single evanescent Bloch mode carries no power. The classical forward-backward
# model handles stopbands with the edge basis instead. Set to None to disable.
BLOCH_EDGE_GUARD_DEG = (10, 30.0)


class FBSolverConfig(BaseModel):
    """
    Settings of the forward-backward Bloch CME solver (``model="bloch"``).

    Convergence of the forward-backward iteration, selection of the mixing
    terms and treatment of the modes close to a stopband edge.
    """

    # Unknown options raise instead of being ignored, so that renamed or removed
    # options are not silently replaced by their defaults.
    model_config = ConfigDict(extra="forbid")

    passes: int = Field(
        default=10, ge=1, description="Maximum number of forward-backward passes."
    )
    kappa_cutoff: float = Field(
        default=0.2,
        gt=0,
        description="Phase-mismatch cutoff used to filter 3WM/4WM terms: a term is"
        " kept when its residual mismatch |dk + G_m| after a reciprocal vector is"
        " below kappa_cutoff times the largest wavenumber of its process, capped"
        " at the zone edge pi / d (the largest wavenumber the supercell-averaged"
        " envelopes resolve), so no kept term advances more than"
        " kappa_cutoff * pi per supercell.",
    )
    convergence_threshold: float = Field(
        default=0.01,
        gt=0,
        description="Relative change below which forward/backward waves are"
        " considered converged.",
    )
    save_all_passes: bool = Field(
        default=False,
        description="Whether to store intermediate results for every pass.",
    )
    m_max: int | Literal["auto"] = Field(
        default=2,
        description="Largest |m| of the reciprocal lattice"
        " vectors G_m = 2 pi m / d whose intracell Fourier components couple"
        " quasi-phase-matched terms. Only terms whose residual mismatch"
        " |dk + G_m| passes kappa_cutoff are kept; 0 keeps cell averages only."
        " 'auto': 1 for every mode, raised to the Bragg"
        " order m (k d = m pi) of a mode that lies inside a stopband or whose"
        " splitting is below twice edge_threshold, at the frequencies (and in the"
        " segments) where it does; a process uses the largest limit among its"
        " modes, so partner terms stay together. The counter-propagating"
        " component of a mode at the m-th edge is reached only through G_m.",
    )
    anderson_depth: int = Field(
        default=10,
        ge=0,
        description="Number of previous"
        " passes used by Anderson acceleration of the forward-backward"
        " iteration (0 = plain fixed-point iteration). With acceleration,"
        " convergence_threshold applies to the relative fixed-point residual"
        " of the backward envelopes.",
    )
    edge_threshold: float | None = Field(
        default=0.10,
        gt=0,
        lt=1,
        description="A mode is close to a stopband edge when its Bloch splitting"
        " |sinh(gamma d)| is below this value. What happens to the modes close"
        " to an edge or inside a stopband is set by simulate_stopband_modes."
        " Modes with an input are never changed. None: no stopband treatment"
        " (native Bloch basis everywhere, which fails at the edges).",
    )
    simulate_stopband_modes: bool = Field(
        default=True,
        description="True: modes without input"
        " close to an edge or inside a stopband are expanded on a fixed Bloch"
        " pair, the lossless pair on the passband side of the nearest edge where"
        " the splitting equals edge_threshold, and their Bragg coupling between"
        " the two directions is solved exactly once per pass (see"
        " twpasolver.cmes.bloch_fb); their processes are kept and conserve the"
        " photon fluxes. False (faster): their processes are switched off, with"
        " a smoothstep weight in the splitting that falls from 1 at"
        " edge_threshold to 0 at the edge on the passband side, and weight 0"
        " throughout the stopband; every process is scaled by the product of the"
        " weights of its modes, so what is kept still conserves the fluxes.",
    )

    @field_validator("m_max")
    @classmethod
    def _check_m_max(cls, v):
        if v != "auto" and v < 0:
            raise ValueError("m_max must be >= 0 or 'auto'")
        return v

    @property
    def m_max_fixed(self) -> int:
        """``m_max``, or ``BLOCH_M_MAX_AUTO_BASE`` when ``m_max='auto'``."""
        return BLOCH_M_MAX_AUTO_BASE if self.m_max == "auto" else int(self.m_max)
