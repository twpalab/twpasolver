"""Tests for twoport module."""

import numpy as np
import pytest

from twpasolver.abcd_matrices import ABCDArray
from twpasolver.mathutils import s2a
from twpasolver.models.rf_functions import inductance, series_impedance_abcd
from twpasolver.twoport import TwoPortCell, TwoPortModel


class RandomModel(TwoPortModel):
    """Generate random normal ABCD matrices."""

    mu: float = 0
    sigma: float = 1

    def single_abcd(self, freqs: np.ndarray) -> ABCDArray:
        """Generate random abcd."""
        return ABCDArray(
            np.random.normal(loc=self.mu, scale=self.sigma, size=(len(freqs), 2, 2))
        )


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
        twoport_cell.Z0 = -1.1


def test_freq_setter(twoport_cell):
    """Test frequency array setter."""
    with pytest.raises(ValueError):
        twoport_cell.freqs = np.array([1, 2])
    with pytest.raises(ValueError):
        twoport_cell.freqs = np.array([1, 2, -0.1])
    with pytest.raises(ValueError):
        twoport_cell.freqs = np.array([[1], [2], [3]])


def test_getitem():
    """Test TwoPortCell slicing."""
    freqs = np.array([1e9, 2e9, 3e9])
    abcd_mat = np.array([[[1, 2], [0.5, 0.5]]] * 3)
    cell = TwoPortCell(freqs, abcd_mat, Z0=50)

    sliced_cell = cell[1:]

    assert np.array_equal(sliced_cell.freqs, freqs[1:])
    assert np.array_equal(sliced_cell.abcd, abcd_mat[1:])
    assert sliced_cell.Z0 == 50


def test_interpolate():
    freqs_base = np.linspace(1, 10, 10)
    freqs_interp = np.linspace(1, 10, 100)
    L_abcd = series_impedance_abcd(inductance(freqs_base, 0.1))
    L_interp = inductance(freqs_interp, 0.1)
    cell_base = TwoPortCell(freqs_base, L_abcd)
    cell_interp_cartesian = cell_base.interpolate(freqs_interp, polar=False)
    cell_interp_polar = cell_base.interpolate(freqs_interp, polar=True)
    assert np.allclose(cell_interp_cartesian.abcd.B, L_interp)
    assert np.allclose(cell_interp_polar.abcd.B, L_interp)


def test_model_update():
    """Test generation of random cell from model."""
    mod = RandomModel()
    mod.update(mu=2, sigma=5)
    assert mod.mu == 2
    assert mod.sigma == 5
    with pytest.raises(RuntimeError):
        mod.update(not_an_attribute=10)


def test_model_get_abcd():
    """Test generation of random cell from model."""
    mod = RandomModel()
    f = np.arange(10, 20, 0.01)
    cell = mod.get_cell(f)
    assert np.array_equal(cell.freqs, f)
    assert np.isclose(np.mean(cell.abcd), mod.mu, atol=0.1)
    assert np.isclose(np.std(cell.abcd), mod.sigma, atol=0.1)
