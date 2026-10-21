"""Configuration of Runge-Kutta solver."""

from __future__ import annotations

from typing import Literal

from pydantic import AliasChoices, ConfigDict, Field, field_validator

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

# Cap on |raw_launch / (gamma * B_launch)| in the boundary reflection update.
# This term is a perturbative correction to Zb; it is only valid
# while it stays small. Near stopband edges with nonlinear interaction active,
# B_launch can pass through a near-node while raw_launch stays finite, driving
# the ratio (and hence the reflection coefficient) outside the physical
# |Gamma| <= 1 regime. Saturating the ratio keeps the correction perturbative
# by construction instead of letting it diverge.
BOUNDARY_CORR_MAX = 0.3


# Harmonic limit of the modes away from stopbands with FBSolverConfig.m_max =
# 'auto' (also used by the models without an automatic choice).
BLOCH_M_MAX_AUTO_BASE = 1

# Bloch no-reflections model: stopband-edge guard (the forward-backward
# models, uniform and segmented, use FBSolverConfig.edge_threshold and
# simulate_stopband_modes instead). At a band edge the
# forward and backward Bloch modes of a mode coalesce (an exceptional point of
# the supercell transfer matrix): the normalisation W = <psi+, psi-> vanishes
# and the projected couplings grow as 1/|sinh(gamma d)|, so the cell-averaged
# equations stop being reliable. The guard reads the normalisation-free
# splitting of the two Bloch modes,
#     e = |sinh(gamma d)| = |lambda+ - lambda-| / 2,
# which is 0 at every edge and grows again on both sides, in the passband and
# inside the gap. Each mode gets a weight that is 0 for e below the first
# value, 1 above the second, with a smoothstep in between; every process is
# scaled by the product of the weights of its modes, so whatever is kept
# conserves energy exactly. Only a narrow window around each edge is switched
# off: couplings of modes deep inside a stopband are kept. Modes with a nonzero
# input amplitude are never attenuated (a warning is logged instead).
# Set to None to disable.
BLOCH_EDGE_GUARD = (0.05, 0.05)

# Bloch no-reflections model only: reactive-angle guard. A forward-only
# description needs every mode to be a power-carrying travelling wave, i.e.
# a real Bloch normalisation W (it is, in every lossless passband,
# independently of the supercell reference plane). Inside a gap W turns
# reactive, a single evanescent Bloch mode carries no power, and forward-only
# processes involving it violate Manley-Rowe (spurious growth or spurious
# loss). Each mode gets a weight that is 1 for |arg W| below the first angle
# (degrees), 0 above the second, with a smoothstep in between, combined with
# BLOCH_EDGE_GUARD. The forward-backward model keeps both Bloch modes and does
# not use it. Set to None to disable.
BLOCH_EDGE_GUARD_DEG = (10, 30.0)


# Bloch no-reflections model: a mixing term is kept only if its phase mismatch
# over one supercell satisfies |dk| * d < BLOCH_CELL_PHASE_CUTOFF. Its couplings
# are the cell average (G = 0 Fourier component) of the intracell profiles,
# which is the right component only while dk is closer to 0 than to any
# reciprocal lattice vector 2 pi m / d, i.e. |dk| d < pi. The forward-backward
# models keep the G != 0 components and use kappa_cutoff relative to the
# process wavenumber capped at the zone edge pi / d instead (see
# FBSolverConfig.kappa_cutoff).
BLOCH_CELL_PHASE_CUTOFF = 3.141592653589793


def configure_solver(atol=None, rtol=None, max_steps=None, first_step=None):
    """
    Configure global solver parameters.

    Args:
        atol (float, optional): Absolute tolerance for the solver
        rtol (float, optional): Relative tolerance for the solver
        max_steps (int, optional): Maximum number of steps (0 = unlimited)
        first_step (float, optional): Initial step size
    """
    global SOLVER_ATOL, SOLVER_RTOL, SOLVER_MAX_STEPS, SOLVER_FIRST_STEP

    if atol is not None:
        SOLVER_ATOL = atol
    if rtol is not None:
        SOLVER_RTOL = rtol
    if max_steps is not None:
        SOLVER_MAX_STEPS = max_steps
    if first_step is not None:
        SOLVER_FIRST_STEP = first_step


def get_solver_config():
    """
    Get current solver configuration.

    Returns:
        dict: Dictionary with current solver parameters
    """
    return {
        "atol": SOLVER_ATOL,
        "rtol": SOLVER_RTOL,
        "max_steps": SOLVER_MAX_STEPS,
        "first_step": SOLVER_FIRST_STEP,
    }


class FBSolverConfig(BaseModel):
    """
    Convergence parameters for the forward-backward CME solver.

    Groups the knobs that control the reflection self-consistency loop
    used by the GENERAL gain model.
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
        " kept when its mismatch is below kappa_cutoff times the largest"
        " wavenumber of its process. Bloch models: residual mismatch |dk + G_m|"
        " after a reciprocal vector, and the reference wavenumber is capped at"
        " the zone edge pi / d (the largest wavenumber the supercell-averaged"
        " envelopes resolve), so no kept term advances more than"
        " kappa_cutoff * pi per supercell.",
    )
    update_rate: float = Field(
        default=1,
        gt=0,
        le=1,
        description="Damping factor for the reflection self-consistency update.",
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
        description="Bloch models only: largest |m| of the reciprocal lattice"
        " vectors G_m = 2 pi m / d whose intracell Fourier components couple"
        " quasi-phase-matched terms. Only terms whose residual mismatch"
        " |dk + G_m| passes kappa_cutoff are kept; 0 keeps cell averages only."
        " 'auto' (forward-backward models): 1 for every mode, raised to the Bragg"
        " order m (k d = m pi) of a mode that lies inside a stopband or whose"
        " splitting is below twice edge_threshold, at the frequencies (and in the"
        " segments) where it does; a process uses the largest limit among its"
        " modes, so partner terms stay together. The counter-propagating"
        " component of a mode at the m-th edge is reached only through G_m.",
    )
    anderson_depth: int = Field(
        default=10,
        ge=0,
        description="Bloch forward-backward solver only: number of previous"
        " passes used by Anderson acceleration of the forward-backward"
        " iteration (0 = plain fixed-point iteration). With acceleration,"
        " convergence_threshold applies to the relative fixed-point residual"
        " of the backward envelopes.",
    )
    edge_threshold: float | None = Field(
        default=0.10,
        gt=0,
        lt=1,
        validation_alias=AliasChoices("edge_threshold", "edge_basis_threshold"),
        description="Bloch forward-backward solvers (uniform and segmented): a"
        " mode is close to a stopband edge when its Bloch splitting"
        " |sinh(gamma d)| is below this value. What happens to the modes close"
        " to an edge or inside a stopband is set by simulate_stopband_modes."
        " Modes with an input are never changed. None: no stopband treatment"
        " (native Bloch basis everywhere, which fails at the edges).",
    )
    simulate_stopband_modes: bool = Field(
        default=True,
        description="Bloch forward-backward solvers. True: modes without input"
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

    def as_solver_args(self) -> tuple:
        """Return values as the positional/keyword tuple used by the numba kernel."""
        return (
            self.passes,
            self.kappa_cutoff,
            self.update_rate,
            self.convergence_threshold,
            self.save_all_passes,
        )
