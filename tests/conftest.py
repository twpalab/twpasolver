"""Definition of shared fixtures."""

import numpy as np
import pytest

from twpasolver.matrices_arrays import ABCDArray, TwoByTwoArray
from twpasolver.models.base_components import Capacitance, Inductance
from twpasolver.models.model_arrays import TWPA
from twpasolver.twoport import TwoPortCell


@pytest.fixture()
def array2x2_numpy():
    """Predefined data for 2x2 matrices arrays."""
    return np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8j]]])


@pytest.fixture()
def tbt_array(array2x2_numpy):
    """Fixture providing an ABCDArray instance with predefined data."""
    return TwoByTwoArray(array2x2_numpy)


@pytest.fixture()
def abcd_array(array2x2_numpy):
    """Fixture providing an ABCDArray instance with predefined data."""
    return ABCDArray(array2x2_numpy)


@pytest.fixture
def twoport_cell():
    """Fixture for simple TwoPortCell."""
    freqs = np.array([1e9, 2e9, 3e9])
    mat = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8j]], [[2, 3 + 1j], [4, 5]]])
    return TwoPortCell(freqs, mat, Z0=50)


@pytest.fixture
def twpa_data():
    """Return data for a TWPA model with LCLf cells."""
    return {
        "Lk": 8.5,
        "t": 13,
        "d": 0,
        "s": 1,
        "nonlinear": 0,
        "repetition": "60-6",
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
        "Is0": 0.14e-6,
        "fmax": 10,
        "fstep": 0.001,
        "f": 0,
        "df": 0.05,
        "dfs": [0.05],
        "fres": 10,
    }


@pytest.fixture
def twpa(twpa_data):
    """Return an instance of a TWPA model."""
    return TWPA(**twpa_data)


@pytest.fixture
def capacitor():
    """Return example capacitance."""
    return Capacitance(C=1e-15)


@pytest.fixture
def inductor():
    """Return example inductor."""
    return Inductance(L=1e-12)
