"""Tests for twoport module."""
import numpy as np
import pytest

from twpasolver.abcd_matrices import ABCDArray
from twpasolver.mathutils import s2a
from twpasolver.twoport import TwoPortCell


def test_init():
    """Test TwoPortCell initialization."""
    freqs = np.array([1e9, 2e9, 3e9])
    abcd_mat = np.array([[[1, 2], [0.5, 0.5]]] * 3)
    Z0 = 50
    cell = TwoPortCell(freqs, abcd_mat, Z0=Z0)

    assert np.array_equal(cell.freqs, freqs)
    assert np.array_equal(cell.abcd, abcd_mat)
    assert isinstance(cell.abcd, ABCDArray)
    assert cell.Z0 == Z0


def test_init_with_s_parameters():
    """Test TwoPortCell initialization from S-parameters."""
    freqs = np.array([1e9, 2e9, 3e9])
    s_mat = np.array([[[0.5, 0.5j], [0.5, 0.5]]] * 3)
    Z0 = 50
    cell = TwoPortCell.from_s_par(freqs, s_mat, Z0=Z0)

    assert np.array_equal(cell.freqs, freqs)
    assert np.allclose(cell.abcd, s2a(s_mat, Z0))
    assert cell.Z0 == Z0


def test_Z0_setter(twoport_cell):
    """Test TwoPortCell Z0 setter."""
    twoport_cell.Z0 = 75
    assert twoport_cell.Z0 == 75

    with pytest.raises(ValueError):
        twoport_cell.Z0 = 0


def test_freq_setter(twoport_cell):
    """Test frequency array setter."""
    with pytest.raises(ValueError):
        twoport_cell.freqs = np.array([1, 2])
    with pytest.raises(ValueError):
        twoport_cell.freqs = np.array([1, 2, -0.1])
    with pytest.raises(ValueError):
        twoport_cell.freqs = np.array([[1], [2], [3]])


def test_matmul(twoport_cell):
    """Test TwoPortCell matrix multiplication."""
    result_cell = twoport_cell @ twoport_cell
    abcd_mat = np.asarray(twoport_cell.abcd)
    expected_abcd = abcd_mat @ abcd_mat
    assert np.allclose(np.asarray(result_cell.abcd), expected_abcd)
    assert result_cell.Z0 == 50


def test_pow(twoport_cell):
    """Test TwoPortCell matrix exponentiation."""
    result_cell = twoport_cell**2
    abcd_mat = np.asarray(twoport_cell.abcd)
    expected_abcd = abcd_mat @ abcd_mat
    assert np.allclose(np.asarray(result_cell.abcd), expected_abcd)
    assert result_cell.Z0 == 50


def test_getitem():
    """Test TwoPortCell slicing."""
    freqs = np.array([1e9, 2e9, 3e9])
    abcd_mat = np.array([[[1, 2], [0.5, 0.5]]] * 3)
    cell = TwoPortCell(freqs, abcd_mat, Z0=50)

    sliced_cell = cell[1:]

    assert np.array_equal(sliced_cell.freqs, freqs[1:])
    assert np.array_equal(sliced_cell.abcd, abcd_mat[1:])
    assert sliced_cell.Z0 == 50
