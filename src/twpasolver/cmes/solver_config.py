"""Configuration of Runge-Kutta solver."""

from __future__ import annotations

from pydantic import Field

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


# Bloch models: stopband-edge guard (all Bloch models). At a band edge the
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

# Bloch forward-backward model (uniform devices): fixed ("edge") Bloch basis
# for weak modes close to a stopband edge. A weak mode without input whose
# splitting |sinh(gamma d)| is below this value is expanded on the lossless
# Bloch pair of the frequency, on the passband side of the nearest edge, where
# the splitting equals the same value. The basis is therefore anchored at the
# switch-over point: there the fixed and native bases coincide and the switch
# is seamless. Its Bragg coupling between the two directions is solved exactly
# once per pass (see twpasolver.cmes.bloch_fb), and the edge guard leaves the
# mode switched on. The value sets two opposite cell-averaging errors: the
# nonlinear source carries 1/W~ with W~ ~ e (large for small e, as in the
# native basis near the edge), the Bragg term of K grows as e. Around 0.1 both
# are small (edge-gain spread 0.005-0.015 dB between 0.10 and 0.12, ~0.13 dB
# between 0.06 and 0.12 on the imskit holotype). It must not be smaller than
# the guard thresholds. Set to None to disable.
BLOCH_EDGE_BASIS = 0.10

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


# Bloch models: a mixing term is kept only if its phase mismatch over one
# supercell satisfies |dk| * d < BLOCH_CELL_PHASE_CUTOFF. The Bloch couplings
# are the cell average (G = 0 Fourier component) of the intracell profiles,
# which is the right component only while dk is closer to 0 than to any
# reciprocal lattice vector 2 pi m / d, i.e. |dk| d < pi. Terms beyond that
# would need the G != 0 components (quasi-phase matching), not implemented.
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

    passes: int = Field(
        default=10, ge=1, description="Maximum number of forward-backward passes."
    )
    kappa_cutoff: float = Field(
        default=0.2,
        gt=0,
        description="Phase-mismatch cutoff used to filter 3WM/4WM terms.",
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
    m_max: int = Field(
        default=2,
        ge=0,
        description="Bloch models only: largest |m| of the reciprocal lattice"
        " vectors G_m = 2 pi m / d whose intracell Fourier components couple"
        " quasi-phase-matched terms. Only terms whose residual mismatch"
        " |dk + G_m| passes kappa_cutoff are kept; 0 keeps cell averages only.",
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

    def as_solver_args(self) -> tuple:
        """Return values as the positional/keyword tuple used by the numba kernel."""
        return (
            self.passes,
            self.kappa_cutoff,
            self.update_rate,
            self.convergence_threshold,
            self.save_all_passes,
        )
