"""
Twpasolver: TWPA Analysis and Simulation Library
================================================

``twpasolver`` is a Python library for analyzing and simulating Traveling Wave
Parametric Amplifiers (TWPAs) by solving systems of Coupled Mode Equations (CMEs).

Key Features
------------

**Extended CME Systems:**
    - Solve CMEs with arbitrary numbers of modes beyond pump-signal-idler
    - Automatic inclusion of pump harmonics and frequency conversion terms
    - Support for 3-wave mixing (3WM) and 4-wave mixing (4WM) processes
    - Rotating Wave Approximation (RWA) term selection

**Mode Management:**
    - ``ModeArray`` class with symbolic frequency dependency tracking
    - Automatic mode relationship propagation using dependency graphs
    - Support for forward and backward propagating modes
    - Parameter interpolation (kappa, gamma, alpha) from base data

**Performance:**
    - Numba-accelerated parallel CME solvers
    - Vectorised 2x2 ABCD matrix arrays

**Gain Models:**
    - SIMPLE: Traditional coupled mode equations of a homogeneous line, forward
      waves only (fast, approximate couplings)
    - BLOCH: Forward-backward coupled mode equations in the Bloch-mode basis
      (exact couplings, reflections, connectors, stopband edges, segmented
      devices)

**Circuit Modeling:**
    - Hierarchical composition of one-port and two-port network elements
    - Pre-built KITWPA cell models (lumped or distributed elements)
    - Custom circuit elements with automatic ABCD matrix computation

**Analysis Tools:**
    - Phase matching analysis with custom mode configurations
    - Gain and bandwidth calculations with pump depletion effects
    - Parameter sweeps with automatic result caching
    - Built-in plotting functions

Core Classes
------------
* :class:`~twpasolver.analysis.TWPAnalysis`: Main analysis engine with extended CME support
* :class:`~twpasolver.modes_rwa.ModeArray`: Mode relationship handler with symbolic frequency propagation
* :class:`~twpasolver.models.twoportarrays.TWPA`: Nonlinear TWPA model with circuit composition
* :class:`~twpasolver.matrices_arrays.ABCDArray`: Optimized 2x2 matrix arrays
* :class:`~twpasolver.twoport.TwoPortCell`: Network parameter container with format conversion

Usage Examples
--------------
See the :ref:`tutorials` section for detailed examples from simple linear component analysis to complex
multi-mode simulations with pump harmonics and frequency conversion processes.
"""

import hashlib
import importlib.metadata as im
import os
import re
from pathlib import Path

__version__ = im.version(__package__)


def _drop_stale_numba_cache(root: Path) -> None:
    """
    Delete the cached Numba kernels under ``root`` when any jitted source changed.

    Numba re-checks only the file that defines a cached kernel, so editing a
    callee in another module (e.g. ``cmes/rk45.py``) would keep the stale kernel.
    """
    jitted = re.compile(r"^@(nb\.)?njit", re.MULTILINE)
    sources = sorted(
        p for p in root.rglob("*.py") if jitted.search(p.read_text(encoding="utf-8"))
    )
    digest = hashlib.sha1(b"".join(p.read_bytes() for p in sources)).hexdigest()
    stamp = root / "__pycache__" / "numba_sources.sha1"
    try:
        if stamp.exists() and stamp.read_text() == digest:
            return
        for src in sources:
            for f in (src.parent / "__pycache__").glob(f"{src.stem}.*.nb[ic]"):
                f.unlink(missing_ok=True)
        stamp.parent.mkdir(exist_ok=True)
        stamp.write_text(digest)
    except OSError:
        pass  # read-only install: Numba caches elsewhere and nothing is edited


# shortcut: a user-set NUMBA_CACHE_DIR is not managed, clear it by hand after
# editing jitted code.
if not os.environ.get("NUMBA_CACHE_DIR"):
    _drop_stale_numba_cache(Path(__file__).parent)

from twpasolver.analysis import TWPAnalysis
from twpasolver.file_utils import read_file, save_to_file
from twpasolver.frequency import Frequencies
from twpasolver.matrices_arrays import ABCDArray
from twpasolver.twoport import TwoPortCell
