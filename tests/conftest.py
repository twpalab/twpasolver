"""Definition of shared fixtures."""

import numpy as np
import pytest

from twpasolver.matrices_arrays import ABCDArray
from twpasolver.models.twpa import TWPA
from twpasolver.twoport import TwoPortCell


@pytest.fixture()
def abcd_array():
    """Fixture providing an ABCDArray instance with predefined data."""
    mat = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8j]]])
    return ABCDArray(mat)


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
        "name": "model_cpw_dartwars_13nm_Lk8.5",
        "Lk": 8.5,
        "t": 13,
        "d": 0,
        "s": 1,
        "nonlinear": 0,
        "repetition": "60-6",
        "cells": [
            {
                "name": "Unloaded cell",
                "l": 102,
                "Z": 50,
                "N": 30,
                "L": 49.9e-12,
                "C": 24.2e-15,
                "Lf": 0.87e-9,
            },
            {
                "name": "Loaded cell",
                "l": 33.5,
                "Z": 80,
                "N": 6,
                "L": 48.6e-12,
                "C": 8.2e-15,
                "Lf": 0.28e-9,
            },
            {
                "name": "Unloaded cell",
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
def twpa_model(twpa_data):
    """Return an instance of a TWPA model."""
    return TWPA(**twpa_data)
