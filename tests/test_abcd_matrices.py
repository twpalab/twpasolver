"""Tests for abcd_matrices module."""
import numpy as np
import pytest

from twpasolver.abcd_matrices import ABCDArray


def test_create_abcd_array():
    """Test creating an ABCDArray instance and verifying its internal data."""
    mat = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8j]]])
    abcd_array = ABCDArray(mat)
    assert np.array_equal(abcd_array._abcd, mat)


def test_invalid_input_shape():
    """Test raising an exception for an invalid input shape."""
    with pytest.raises(ValueError, match="Input must be array of 2x2 matrices."):
        wrong_mat = np.array([[[1, 2, 3], [4, 5, 6]]])
        ABCDArray(wrong_mat)


def test_getitem(abcd_array):
    """Test accessing values using the __getitem__ method."""
    assert np.array_equal(abcd_array[0], np.array([[1, 2], [3, 4]]))
    assert np.array_equal(abcd_array[1][0], np.array([5, 6]))
    sliced = abcd_array[:-1]
    assert isinstance(sliced, ABCDArray)
    assert np.array_equal(np.asarray(sliced), np.asarray(abcd_array[:-1]))


def test_setitem(abcd_array):
    """Test setting values using the __setitem__ method."""
    abcd_array[1][0][0] = 10
    expected_result = ABCDArray(np.array([[[1, 2], [3, 4]], [[10, 6], [7, 8j]]]))
    assert np.array_equal(abcd_array._abcd, expected_result._abcd)


def test_shape_and_len(abcd_array):
    """Test getting the shape and length properties."""
    assert abcd_array.shape == (2, 2, 2)
    assert abcd_array.len == 2


def test_A_property(abcd_array):
    """Test accessing the A property."""
    assert np.array_equal(abcd_array.A, np.array([1, 5]))


def test_A_property_setter(abcd_array):
    """Test setting values using the A property setter."""
    abcd_array.A = np.array([10, 9])
    expected_result = ABCDArray(np.array([[[10, 2], [3, 4]], [[9, 6], [7, 8j]]]))
    assert np.array_equal(abcd_array._abcd, expected_result._abcd)


def test_pow_and_mul(abcd_array):
    """Test exponentiation and matrix multiplication."""
    pow = np.asarray(abcd_array**2)
    mul = np.asarray(abcd_array @ abcd_array)
    mat = np.asarray(abcd_array)
    assert np.array_equal(pow, mat @ mat)
    assert np.array_equal(mul, mat @ mat)