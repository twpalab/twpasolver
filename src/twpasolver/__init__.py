"""
``twpasolver`` package.

Twpasolver: TWPA Analysis and Simulation Library
================================================

``twpasolver`` is a Python library for analyzing and simulating Traveling Wave
Parametric Amplifiers (TWPAs) by solving systems of Coupled Mode Equations (CMEs).

Key Features
------------

**Extended CME Systems:**
    - Solve CMEs with arbitrary numbers of modes beyond pump-signal-idler
    - Automatic inclusion of pump harmonics, frequency conversion terms, and harmonics
    - Support for 3-wave mixing (3WM) and 4-wave mixing (4WM) processes
    - Rotating Wave Approximation (RWA) term selection

**Mode Management:**
    - ``ModeArray`` class with symbolic frequency dependency tracking
    - Automatic mode relationship propagation using dependency graphs
    - Support for forward and backward propagating modes
    - Parameter interpolation (kappa, gamma, alpha) from base data

**Performance:**
    - Numba-accelerated parallel CME solvers
    - Efficient matrix operations for 2x2 ABCD matrices
    - Frequency array processing with caching
    - Parameter sweeps with result caching

**Gain Models:**
    - MINIMAL_3WM: Basic three-wave mixing model with only pump, signal and idler
    - GENERAL_IDEAL: Extended modes without losses or reflections
    - GENERAL_LOSS_ONLY: Extended modes with losses
    - GENERAL: Full model with losses and reflections

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

import importlib.metadata as im

__version__ = im.version(__package__)

from twpasolver.analysis import TWPAnalysis
from twpasolver.file_utils import read_file, save_to_file
from twpasolver.frequency import Frequencies
from twpasolver.matrices_arrays import ABCDArray
from twpasolver.twoport import TwoPortCell
