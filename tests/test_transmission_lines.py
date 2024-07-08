"""Test transmission line module."""

import numpy as np

from twpasolver.models.compose import compose
from twpasolver.models.transmission_lines import LosslessTL


def test_from_z0_vp(lossless_line):
    """Test initialization from impedance and phase velocity."""
    line_copy = LosslessTL.from_z_vp(
        lossless_line.Z0, lossless_line.vp, lossless_line.l
    )
    assert np.isclose(line_copy.C, lossless_line.C)
    assert np.isclose(line_copy.L, lossless_line.L)
    assert np.isclose(line_copy.l, lossless_line.l)


def test_composed_line(lossless_line):
    """Test equality between various methods of creating longer lines."""
    freqs = np.arange(1e9, 10e9, 5e7)
    double_length_compose = compose(lossless_line, lossless_line)
    double_length = lossless_line.model_copy()
    double_length.l = lossless_line.l * 2
    lossless_line.N = 2
    assert np.allclose(double_length.get_abcd(freqs), lossless_line.get_abcd(freqs))
    assert np.allclose(
        double_length_compose.get_abcd(freqs), lossless_line.get_abcd(freqs)
    )
