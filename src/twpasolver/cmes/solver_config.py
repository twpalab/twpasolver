"""Settings of the RK45 integrator and of the forward-backward Bloch solver."""

from __future__ import annotations

from typing import Literal

from pydantic import ConfigDict, Field, field_validator

from twpasolver.basemodel import BaseModel


class RKSolverConfig(BaseModel):
    """
    Settings of the adaptive RK45 integrator used by every CME kernel.

    They are passed to the Numba kernels as arguments, so changing them takes
    effect at once (no recompilation, no stale cache).
    """

    model_config = ConfigDict(extra="forbid")

    atol: float = Field(default=1e-12, gt=0, description="Absolute tolerance.")
    rtol: float = Field(default=1e-8, gt=0, description="Relative tolerance.")
    first_step: float = Field(
        default=10, gt=0, description="Initial step, in unit cells."
    )
    max_steps_per_cell: float = Field(
        default=10,
        ge=0,
        description="Cap on the accepted steps per unit cell of the integration"
        " interval (device or segment length): ceil(length * max_steps_per_cell)"
        " + 1. The CME envelopes vary slowly on the scale of a cell, so a run that"
        " needs more is running away; it is stopped and its unreached points are"
        " NaN (reported as a warning). 0 disables the cap.",
    )

    def kernel_args(self) -> tuple[float, float, float, float]:
        """``(atol, rtol, first_step, max_steps_per_cell)`` for the kernels."""
        return (
            float(self.atol),
            float(self.rtol),
            float(self.first_step),
            float(self.max_steps_per_cell),
        )


# Harmonic limit of the modes away from stopbands with BlochSolverConfig.m_max =
# 'auto'.
BLOCH_M_MAX_AUTO_BASE = 1


class BlochSolverConfig(BaseModel):
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
        ge=1,
        description="Number of previous"
        " passes used by Anderson acceleration of the forward-backward"
        " iteration. convergence_threshold applies to the relative fixed-point"
        " residual of the backward envelopes; a point whose residual turns"
        " NaN/inf is stopped at once and reported as not converged.",
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
