"""Configuration of Runge-Kutta solver."""

from __future__ import annotations

from pydantic import Field

from twpasolver.basemodel import BaseModel

# Global solver configuration
SOLVER_ATOL = 1e-12
SOLVER_RTOL = 1e-8
SOLVER_MAX_STEPS = 0
SOLVER_RK_METHOD = 1
SOLVER_FIRST_STEP = 10


def configure_solver(
    atol=None, rtol=None, max_steps=None, rk_method=None, first_step=None
):
    """
    Configure global solver parameters.

    Args:
        atol (float, optional): Absolute tolerance for the solver
        rtol (float, optional): Relative tolerance for the solver
        max_steps (int, optional): Maximum number of steps (0 = unlimited)
        rk_method (int, optional): Runge-Kutta method (1=RK45, 2=DOP853, etc.)
        first_step (float, optional): Initial step size
    """
    global SOLVER_ATOL, SOLVER_RTOL, SOLVER_MAX_STEPS, SOLVER_RK_METHOD, SOLVER_FIRST_STEP

    if atol is not None:
        SOLVER_ATOL = atol
    if rtol is not None:
        SOLVER_RTOL = rtol
    if max_steps is not None:
        SOLVER_MAX_STEPS = max_steps
    if rk_method is not None:
        SOLVER_RK_METHOD = rk_method
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
        "rk_method": SOLVER_RK_METHOD,
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
        default=0.95,
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

    def as_solver_args(self) -> tuple:
        """Return values as the positional/keyword tuple used by the numba kernel."""
        return (
            self.passes,
            self.kappa_cutoff,
            self.update_rate,
            self.convergence_threshold,
            self.save_all_passes,
        )
