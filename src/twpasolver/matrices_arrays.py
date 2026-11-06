"""ABCD matrices array module."""

from typing import Any

import numpy as np
from typing_extensions import Self

from twpasolver.mathutils import matmul_2x2, matpow_2x2


class TwoByTwoArray:
    """A class representing an array of 2x2 matrices."""

    def __init__(self, mat: np.ndarray):
        """
        Initialize the TwoByTwoArray instance.

        Args:
            mat (numpy.ndarray): Input array of 2x2 matrices or [x11,x12,x21,x22], where each element is a 1D list.
        """
        mat = np.asarray(mat)
        # Transform the input to array of 2x2 matrices
        if len(mat.shape) == 2 and len(mat) == 4:
            mat = mat.reshape(2, 2, mat.shape[1]).transpose(2, 0, 1)
        # Check if the array has now the correct shape
        if len(mat.shape) != 3 or mat.shape[-2:] != (2, 2):
            raise ValueError(
                "Input must be array of 2x2 matrices or [A, B, C, D], where each element is a list."
            )
        self._matarray = mat

    def __repr__(self):
        """
        Return a string representation of the TwoByTwoArray.

        Returns:
            str: String representation of the TwoByTwoArray.
        """
        return f"{self.__class__.__name__}({self._matarray})"

    def __array__(self, dtype=None, copy=None):
        """Convert the TwoByTwoArray to a numpy array."""
        return np.array(self._matarray, dtype=dtype, copy=copy)

    def __getitem__(self, indices) -> Self | np.ndarray:
        """
        Get value at indices.

        Args:
            indices: Indices to access the array.

        Returns:
            TwoByTwoArray | numpy.ndarray: Value at the specified indices.
        """
        if isinstance(indices, slice):
            return self.__class__(self._matarray[indices])
        return self._matarray[indices]

    def __setitem__(self, indices, val: float | int):
        """
        Set value at indices.

        Args:
            indices: Indices to set the value.
            val (float | int): Value to set at the specified indices.
        """
        self._matarray[indices] = val

    @property
    def shape(self):
        """Shape of the internal array."""
        return self._matarray.shape

    @property
    def len(self):
        """Length of the internal array."""
        return self._matarray.shape[0]


class _Element:
    """One element of every 2x2 matrix, as a 1-D array; settable."""

    def __init__(self, i: int, k: int):
        self.i, self.k = i, k

    def __set_name__(self, owner, name: str):
        self.__doc__ = f"{name} (element ({self.i},{self.k})) of the 2x2 matrices."

    def __get__(self, obj, objtype=None) -> Any:
        if obj is None:
            return self
        return obj._matarray[:, self.i, self.k]

    def __set__(self, obj, values: np.ndarray):
        values = np.asarray(values)
        if values.ndim != 1 or len(values) != obj.len:
            raise ValueError(f"Must provide a 1-D array of length {obj.len}")
        obj._matarray[:, self.i, self.k] = values


class ABCDArray(TwoByTwoArray):
    """A class representing an array of ABCD matrices."""

    def __matmul__(self, other: Self) -> Self:
        """
        Efficient matrix multiplication with another ABCDArray.

        Args:
            other (ABCDArray): Another ABCDArray for matrix multiplication.

        Returns:
            ABCDArray: Result of the matrix multiplication.
        """
        return self.__class__(matmul_2x2(self._matarray, other._matarray))

    def __pow__(self, exponent: int) -> Self:
        """
        Efficient matrix exponentiation of the ABCDArray.

        Args:
            exponent (int): The exponent to raise the ABCDArray to.

        Returns:
            ABCDArray: Result of raising the ABCDArray to the specified power.
        """
        return self.__class__(matpow_2x2(self._matarray, exponent))

    A = _Element(0, 0)
    B = _Element(0, 1)
    C = _Element(1, 0)
    D = _Element(1, 1)


class SMatrixArray(TwoByTwoArray):
    """A class representing an array of two-ports S parameters matrices."""

    S11 = _Element(0, 0)
    S12 = _Element(0, 1)
    S21 = _Element(1, 0)
    S22 = _Element(1, 1)


class ZMatrixArray(TwoByTwoArray):
    """A class representing an array of two-ports Z parameters matrices."""

    Z11 = _Element(0, 0)
    Z12 = _Element(0, 1)
    Z21 = _Element(1, 0)
    Z22 = _Element(1, 1)


def abcd_identity(N_abcd: int) -> ABCDArray:
    """
    Get abcd array of identity matrices.

    Args:
        N_abcd (int): Number of matrices in array.

    Returns:
        ABCDArray: Array of identity matrices.
    """
    mat = np.zeros((N_abcd, 2, 2), dtype=np.complex128)
    mat[:, 0, 0] = 1.0
    mat[:, 1, 1] = 1.0
    return ABCDArray(mat)
