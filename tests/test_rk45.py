"""The RK45 kernel returns its dense output, not a linear interpolation of the steps."""

import numba as nb
import numpy as np

from twpasolver.cmes.rk45 import rk45_solve


@nb.njit
def _rotate(x, y, k):
    return 1j * k * y - 0.01 * y


@nb.njit
def _solve(t_span, y0, t_eval, k):
    return rk45_solve(
        _rotate, t_span, y0, t_eval, 0.1, args=(k,), rtol=1e-9, atol=1e-14
    )


def test_dense_output_between_steps():
    y0 = np.array([1.0 + 0j, 0.5j])
    for t_eval in (np.linspace(0.0, 50.0, 23), np.linspace(50.0, 0.0, 23)):
        span = (t_eval[0], t_eval[-1])
        y = _solve(span, y0, t_eval, 0.7)
        exact = y0[:, None] * np.exp((0.7j - 0.01) * (t_eval - t_eval[0]))[None, :]
        assert np.abs(y - exact).max() < 1e-7
