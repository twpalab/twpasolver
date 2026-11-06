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


def _warm_analysis(twpa: TWPA, label: str, models: tuple, verbose: bool) -> None:
    if verbose:
        print(
            f"twpasolver warmup: compiling CME kernels for the {label} device path..."
        )
    analysis = TWPAnalysis(twpa=twpa, f_arange=_F_ARANGE)
    analysis.update_base_data()
    with warnings.catch_warnings():
        # No backward input is injected, so the backward-referenced gains are
        # 0/0; harmless for a call that only triggers compilation.
        warnings.simplefilter("ignore", RuntimeWarning)
        for model in models:
            analysis.gain(
                _SIGNAL_FREQS, model=model, fb_config=FBSolverConfig(passes=1)
            )


def warmup(verbose: bool = True) -> None:
    """Trigger and disk-cache compilation of the CME solver kernels.

    ``simple`` has one kernel for uniform and one for segmented devices;
    ``bloch`` has a single kernel (a uniform device is one segment). Safe to
    run repeatedly: once the cache is populated, later runs finish in seconds.
    """
    uniform = TWPA.model_validate(_TWPA_KWARGS)
    _warm_analysis(uniform, "uniform", (GainModel.SIMPLE, GainModel.BLOCH), verbose)
    segmented = TWPA(cells=[uniform, uniform.model_copy(deep=True)], Ip0=250e-6)
    _warm_analysis(segmented, "segmented", (GainModel.SIMPLE,), verbose)
    if verbose:
        print("twpasolver warmup: done, Numba cache populated.")


def main() -> None:
    """Entry point for the ``twpasolver-warmup`` console script."""
    warmup()


if __name__ == "__main__":
    main()
