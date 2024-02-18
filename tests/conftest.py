"""Definition of shared fixtures."""
import numpy as np
import pytest

from twpasolver.abcd_matrices import ABCDArray


@pytest.fixture()
def abcd_array():
    """Fixture providing an ABCDArray instance with predefined data."""
    mat = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8j]]])
    return ABCDArray(mat)
