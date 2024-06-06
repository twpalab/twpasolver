"""Tests for matrices_arrays module."""

import numpy as np
import pytest

from twpasolver.matrices_arrays import ABCDArray, TwoByTwoArray


def test_create_matarray_array():
    """Test creating an TwoByTwoArray instance and verifying its internal data."""
    mat = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8j]]])
    tbt_array = TwoByTwoArray(mat)
    assert np.array_equal(tbt_array._matarray, mat)


def test_create_matarray_array_transposed(tbt_array):
    """Test creating an TwoByTwoArray instance from [A,B,C,D] structure."""
    A = [1, 5]
    B = [2, 6]
    C = [3, 7]
    D = [4, 8j]
    mat = np.array([A, B, C, D])
    tbt_array_from_transpose = TwoByTwoArray(mat)
    assert np.array_equal(tbt_array._matarray, tbt_array_from_transpose._matarray)


def test_invalid_input_shape():
    """Test raising an exception for an invalid input shape."""
    with pytest.raises(ValueError, match="Input must be array of 2x2 matrices."):
        wrong_mat = np.array([[[1, 2, 3], [4, 5, 6]]])
        TwoByTwoArray(wrong_mat)


def test_getitem(tbt_array):
    """Test accessing values using the __getitem__ method."""
    assert np.array_equal(tbt_array[0], np.array([[1, 2], [3, 4]]))
    assert np.array_equal(tbt_array[1][0], np.array([5, 6]))
    sliced = tbt_array[:-1]
    assert isinstance(sliced, TwoByTwoArray)
    assert np.array_equal(np.asarray(sliced), np.asarray(tbt_array[:-1]))


def test_setitem(tbt_array):
    """Test setting values using the __setitem__ method."""
    tbt_array[1][0][0] = 10
    expected_result = TwoByTwoArray(np.array([[[1, 2], [3, 4]], [[10, 6], [7, 8j]]]))
    assert np.array_equal(tbt_array._matarray, expected_result._matarray)


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
    assert np.array_equal(abcd_array._matarray, expected_result._matarray)


def test_pow_and_mul(abcd_array):
    """Test exponentiation and matrix multiplication."""
    pow = np.asarray(abcd_array**2)
    mul = np.asarray(abcd_array @ abcd_array)
    mat = np.asarray(abcd_array)
    assert np.array_equal(pow, mat @ mat)
    assert np.array_equal(mul, mat @ mat)
