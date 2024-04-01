"""Definition of shared fixtures."""


import numpy as np
import pytest

from twpasolver.abcd_matrices import ABCDArray
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
