"""Pre-compile and disk-cache the package's Numba CME solver kernels.

Numba's ``@njit(cache=True)`` only writes a cache entry once a function has
actually been called with a given argument-type signature, the first call
in a fresh install (or a fresh venv/container) still pays full compile time
(minutes, for the kernels in :mod:`twpasolver.cmes`). Run this module once
right after installing twpasolver (``twpasolver-warmup``) to pay that cost
up front instead of on first real use.

This is not literally "compile at install time", pip/Poetry have no
supported post-install hook, it's the practical equivalent: a one-command
step you run after installing (locally, in a Dockerfile, in CI/deploy
setup).
"""

from __future__ import annotations

import warnings

import numpy as np

from twpasolver.analysis import GainModel, TWPAnalysis
from twpasolver.cmes.solver_config import FBSolverConfig
from twpasolver.models.twoportarrays import TWPA

# Same cell physics as tests/conftest.py's `twpa` fixture, an arbitrary but
# known-working device, just used here to exercise every solve path.
_TWPA_KWARGS = {
    "cells": [
        {
            "name": "LCLfBaseCell",
            "l": 102,
            "Z": 50,
            "N": 30,
            "L": 49.9e-12,
            "C": 24.2e-15,
            "Lf": 0.87e-9,
        },
        {
            "name": "LCLfBaseCell",
            "l": 33.5,
            "Z": 80,
            "N": 6,
            "L": 48.6e-12,
            "C": 8.2e-15,
            "Lf": 0.28e-9,
        },
        {
            "name": "LCLfBaseCell",
            "l": 102,
            "Z": 50,
            "N": 30,
            "L": 49.9e-12,
            "C": 24.2e-15,
            "Lf": 0.87e-9,
        },
    ],
    "N": 523,
    "Istar": 6.3e-3,
    "Idc": 1e-3,
    "Ip0": 250e-6,
}
_F_ARANGE = (0.05, 20, 1e-3)
_SIGNAL_FREQS = np.array([2.0, 3.0])


def _warm_analysis(twpa: TWPA, label: str, verbose: bool) -> None:
    if verbose:
        print(
            f"twpasolver warmup: compiling CME kernels for the {label} device path..."
        )
    analysis = TWPAnalysis(twpa=twpa, f_arange=_F_ARANGE)
    analysis.update_base_data()
    with warnings.catch_warnings():
        # The default zero backward-wave initial condition used here (no
        # initial_amplitudes_bwd override) makes analysis.py's bwd-gain
        # ratios divide by zero -- harmless for a warmup call that only
        # cares about triggering compilation, not the returned gain values.
        warnings.simplefilter("ignore", RuntimeWarning)
        # Triggers _no_reflections_solve_{uniform,segmented} (ndim of the
        # gammas array picks uniform vs segmented, driven by `twpa`'s
        # topology).
        analysis.gain(_SIGNAL_FREQS, model=GainModel.GENERAL_NO_REFLECTIONS)
        # Triggers _general_cmes_solve_fb_cutoff_{uniform,segmented}.
        analysis.gain(
            _SIGNAL_FREQS, model=GainModel.GENERAL, fb_config=FBSolverConfig(passes=1)
        )


def warmup(verbose: bool = True) -> None:
    """Trigger and disk-cache compilation of all 4 CME solver njit kernels.

    Runs both gain models (no-reflections and general/forward-backward) on
    both a uniform and a segmented TWPA -- the (model) x (device topology)
    combinations that select between ``_no_reflections_solve_{uniform,
    segmented}`` and ``_general_cmes_solve_fb_cutoff_{uniform,segmented}``
    in :mod:`twpasolver.cmes`. Safe to run repeatedly: once the cache is
    populated, subsequent runs hit it and finish in a few seconds.
    """
    _warm_analysis(TWPA(**_TWPA_KWARGS), "uniform", verbose)
    segmented = TWPA(cells=[TWPA(**_TWPA_KWARGS), TWPA(**_TWPA_KWARGS)], Ip0=250e-6)
    _warm_analysis(segmented, "segmented", verbose)
    if verbose:
        print("twpasolver warmup: done, Numba cache populated.")


def main() -> None:
    """Entry point for the ``twpasolver-warmup`` console script."""
    warmup()


if __name__ == "__main__":
    main()
