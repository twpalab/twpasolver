"""Test frequency list model."""

import numpy as np

from twpasolver.frequency import Frequencies


def test_frequencies_init(frequency_list):
    """Test init and get of simple frequencies."""
    fspan = Frequencies(f_list=frequency_list)
    assert np.array_equal(np.array(frequency_list) * 1e9, fspan.f)


def test_frequencies_init_arange(frequency_arange):
    """Test init and get of simple frequencies from arange."""
    fspan = Frequencies(f_arange=frequency_arange)
    fspan.f_list = [1, 2]
    assert np.array_equal(np.arange(*frequency_arange) * 1e9, fspan.f)


def test_frequencies_omega(frequency_list):
    """Test init and get of simple frequencies."""
    fspan = Frequencies(f_list=frequency_list)
    assert np.array_equal(np.array(frequency_list) * 1e9 * 2 * np.pi, fspan.omega)
