"""Test for twpasolver.models.oneport module."""

import numpy as np
import pytest

from twpasolver.models.oneport import OnePortModel


def test_one_port_model_Z_abstractmethod():
    """
    Test that the Z method in OnePortModel is an abstract method
    and cannot be instantiated directly.
    """
    with pytest.raises(TypeError):
        OnePortModel()


def test_one_port_model_Y(resistance):
    """Test the Y method of OnePortModel."""
    freqs = np.array([1e9, 2e9, 3e9])
    expected_admittance = 1 / resistance.Z(freqs)
    np.testing.assert_array_almost_equal(resistance.Y(freqs), expected_admittance)


def test_one_port_model_single_abcd_series_impedance(resistance):
    """Test the single_abcd method with series impedance."""
    freqs = np.array([1e9, 2e9, 3e9])
    abcd = resistance.single_abcd(freqs)
    assert np.array_equal(abcd.B, resistance.Z(freqs))


def test_one_port_model_single_abcd_parallel_admittance(resistance):
    """Test the single_abcd method with parallel admittance."""
    resistance.twoport_parallel = True
    freqs = np.array([1e9, 2e9, 3e9])
    abcd = resistance.single_abcd(freqs)
    assert np.array_equal(abcd.C, resistance.Y(freqs))


def test_one_port_array_Z_parallel(one_port_array):
    """Test the Z method of OnePortArray with parallel connection."""
    one_port_array.parallel = True
    freqs = np.array([1e9, 2e9, 3e9])
    expected_impedance = 1 / np.sum([c.Y(freqs) for c in one_port_array.cells], axis=0)
    np.testing.assert_array_almost_equal(one_port_array.Z(freqs), expected_impedance)


def test_one_port_array_Z_series(one_port_array):
    """Test the Z method of OnePortArray with series connection."""
    one_port_array.parallel = False
    freqs = np.array([1e9, 2e9, 3e9])
    expected_impedance = np.sum([c.Z(freqs) for c in one_port_array.cells], axis=0)
    np.testing.assert_array_almost_equal(one_port_array.Z(freqs), expected_impedance)


def test_resistance_Z(resistance):
    """Test the Z method of Resistance."""
    freqs = np.array([1e9, 2e9, 3e9])
    expected_impedance = np.full_like(freqs, resistance.R)
    np.testing.assert_array_almost_equal(resistance.Z(freqs), expected_impedance)
