"""Tests for mathutils module."""
import numpy as np
import pytest

from twpasolver.mathutils import matmul_2x2, matpow_2x2, matpow_2x2_recursive

# Replace 'your_module' with the actual name of the module where your functions are defined

# Test data
matrices_a = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
matrices_b = np.array([[[2.0, 0.0], [1.0, 2.0]], [[-1.0, 2.0], [0.0, -1.0]]])
exponent = 3


def test_matmul_2x2():
    """Test matmul_2x2 function."""
    result = matmul_2x2(matrices_a, matrices_b)
    expected_result = np.array([[[4.0, 4.0], [10.0, 8.0]], [[-5.0, 4], [-7.0, 6.0]]])
    assert np.allclose(result, expected_result)


def test_matmul_2x2_shape():
    """Test shape assertion in matmul_2x2 function."""
    with pytest.raises(AssertionError):
        matmul_2x2(matrices_a, matrices_a[:, :, :-1])


def test_matpow_2x2():
    """Test matpow_2x2 function."""
    result = matpow_2x2(matrices_b, exponent)
    expected_result = matrices_b @ matrices_b @ matrices_b
    assert np.allclose(result, expected_result)


def test_matpow_2x2_exponent():
    """Test exponent value assertion in matpow_2x2 function."""
    with pytest.raises(AssertionError):
        matpow_2x2(matrices_a, 1)


def test_matpow_2x2_shape():
    """Test shape assertion in matpow_2x2 function."""
    with pytest.raises(AssertionError):
        matpow_2x2(matrices_a[:, :, :-1], exponent)


def test_matpow_2x2_recursive():
    """Test matpow_2x2_recursive function."""
    result = matpow_2x2_recursive(matrices_a, exponent)
    expected_result = matrices_a @ matrices_a @ matrices_a
    assert np.allclose(result, expected_result)
    result = matpow_2x2_recursive(matrices_a, 4)
    expected_result = matrices_a @ matrices_a @ matrices_a @ matrices_a
    assert np.allclose(result, expected_result)


def test_matpow_2x2_recursive_exponent():
    """Test exponent value assertion in matpow_2x2_recursive function."""
    with pytest.raises(AssertionError):
        matpow_2x2(matrices_a, 0)
    assert (matpow_2x2_recursive(matrices_b, 1) == matrices_b).all()


def test_matpow_2x2_recursive_shape():
    """Test shape assertion in matpow_2x2_recursive function."""
    with pytest.raises(AssertionError):
        matpow_2x2_recursive(matrices_a[:, :, :-1], exponent)


def test_matpow_same():
    """Test same behaviour between two matpow methods."""
    res = matpow_2x2(matrices_a, 8)
    resrec = matpow_2x2_recursive(matrices_a, 8)
    assert np.allclose(res, resrec)
