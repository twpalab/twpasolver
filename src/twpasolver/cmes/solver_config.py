"""Configuration of Runge-Kutta solver."""

# Global solver configuration
SOLVER_ATOL = 1e-14
SOLVER_RTOL = 1e-10
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
