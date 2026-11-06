"""Pre-compile and disk-cache the package's Numba CME solver kernels.

Numba's ``@njit(cache=True)`` writes a cache entry only once a function has
been called with a given signature, so the first call in a fresh environment
pays the full compile time (minutes, for the kernels in
:mod:`twpasolver.cmes`). Run ``twpasolver-warmup`` once after installing
(locally, in a Dockerfile, in CI) to pay that cost up front; pip/Poetry have
no post-install hook to do it automatically.
"""

from __future__ import annotations

import logging
import warnings

import numpy as np

from twpasolver.analysis import GainModel, TWPAnalysis
from twpasolver.cmes.solver_config import BlochSolverConfig
from twpasolver.logger import log
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
_PUMP = 12.0


def _warm_analysis(twpa: TWPA, label: str, models: tuple, verbose: bool) -> None:
    if verbose:
        print(
            f"twpasolver warmup: compiling CME kernels for the {label} device path..."
        )
    analysis = TWPAnalysis(twpa=twpa, freqs=_F_ARANGE)
    level = log.level
    # One pass only compiles the kernels; its "did not converge" warning is noise.
    log.setLevel(logging.ERROR)
    try:
        with warnings.catch_warnings():
            # No backward input is injected, so the backward-referenced gains are
            # 0/0; harmless for a call that only triggers compilation.
            warnings.simplefilter("ignore", RuntimeWarning)
            for model in models:
                analysis.gain(
                    _SIGNAL_FREQS,
                    pump=_PUMP,
                    model=model,
                    bloch_config=BlochSolverConfig(passes=1),
                )
    finally:
        log.setLevel(level)


def warmup(verbose: bool = True) -> None:
    """Trigger and disk-cache compilation of the CME solver kernels.

    Each model has a single kernel (a uniform device is one segment). Safe to
    run repeatedly: once the cache is populated, later runs finish in seconds.
    """
    uniform = TWPA.model_validate(_TWPA_KWARGS)
    _warm_analysis(uniform, "uniform", (GainModel.SIMPLE, GainModel.BLOCH), verbose)
    if verbose:
        print("twpasolver warmup: done, Numba cache populated.")


def main() -> None:
    """Entry point for the ``twpasolver-warmup`` console script."""
    warmup()


if __name__ == "__main__":
    main()
