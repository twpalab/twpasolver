"""ABCD matrices array module."""

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

    def __array__(self):
        """Convert the TwoByTwoArray to a numpy array."""
        return self._matarray

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

    def _get_parameter(self, i: int, k: int):
        """
        Get the specified parameter of the 2x2 matrices.

        Args:
            i (int): Row index (0 or 1).
            k (int): Column index (0 or 1).

        Returns:
            numpy.ndarray: The specified parameter of the 2x2 matrices.
        """
        return self._matarray[:, i, k]

    def _set_parameter(self, i: int, k: int, values: np.ndarray):
        """
        Set the specified parameter of the 2x2 matrices.

        Args:
            i (int): Row index (0 or 1).
            k (int): Column index (0 or 1).
            values (numpy.ndarray): Values to set for the specified parameter.
        """
        if values.ndim != 1 and len(values) != self.len:
            raise ValueError(f"Must provide a 1-D array of length {self.len}")
        self._matarray[:, i, k] = np.asarray(values)


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

    @property
    def A(self) -> np.ndarray:
        """A parameter (element (0,0)) of the 2x2 matrices."""
        return self._get_parameter(0, 0)

    @A.setter
    def A(self, value: np.ndarray):
        """
        Setter for the A parameter.

        Args:
            value (numpy.ndarray): Values to set for the A parameter.
        """
        self._set_parameter(0, 0, value)

    @property
    def B(self) -> np.ndarray:
        """B parameter (element (0,1)) of the 2x2 matrices."""
        return self._get_parameter(0, 1)

    @B.setter
    def B(self, value: np.ndarray):
        """
        Setter for the B parameter.

        Args:
            value (numpy.ndarray): Values to set for the B parameter.
        """
        self._set_parameter(0, 1, value)

    @property
    def C(self) -> np.ndarray:
        """C parameter (element (1,0)) of the 2x2 matrices."""
        return self._get_parameter(1, 0)

    @C.setter
    def C(self, value: np.ndarray):
        """
        Setter for the C parameter.

        Args:
            value (numpy.ndarray): Values to set for the C parameter.
        """
        self._set_parameter(1, 0, value)

    @property
    def D(self) -> np.ndarray:
        """D parameter (element (1,1)) of the 2x2 matrices."""
        return self._get_parameter(1, 1)

    @D.setter
    def D(self, value: np.ndarray):
        """
        Setter for the D parameter.

        Args:
            value (numpy.ndarray): Values to set for the D parameter.
        """
        self._set_parameter(1, 1, value)


class SMatrixArray(TwoByTwoArray):
    """A class representing an array of two-ports S parameters matrices."""

    @property
    def S11(self) -> np.ndarray:
        """S11 parameter (element (0,0)) of the 2x2 matrices."""
        return self._get_parameter(0, 0)

    @S11.setter
    def S11(self, value: np.ndarray):
        """
        Setter for the S11 parameter.

        Args:
            value (numpy.ndarray): Values to set for the S11 parameter.
        """
        self._set_parameter(0, 0, value)

    @property
    def S12(self) -> np.ndarray:
        """S12 parameter (element (0,1)) of the 2x2 matrices."""
        return self._get_parameter(0, 1)

    @S12.setter
    def S12(self, value: np.ndarray):
        """
        Setter for the S12 parameter.

        Args:
            value (numpy.ndarray): Values to set for the S12 parameter.
        """
        self._set_parameter(0, 1, value)

    @property
    def S21(self) -> np.ndarray:
        """S21 parameter (element (1,0)) of the 2x2 matrices."""
        return self._get_parameter(1, 0)

    @S21.setter
    def S21(self, value: np.ndarray):
        """
        Setter for the S21 parameter.

        Args:
            value (numpy.ndarray): Values to set for the S21 parameter.
        """
        self._set_parameter(1, 0, value)

    @property
    def S22(self) -> np.ndarray:
        """S22 parameter (element (1,1)) of the 2x2 matrices."""
        return self._get_parameter(1, 1)

    @S22.setter
    def S22(self, value: np.ndarray):
        """
        Setter for the S22 parameter.

        Args:
            value (numpy.ndarray): Values to set for the S22 parameter.
        """
        self._set_parameter(1, 1, value)


class ZMatrixArray(TwoByTwoArray):
    """A class representing an array of two-ports Z parameters matrices."""

    @property
    def Z11(self) -> np.ndarray:
        """Z11 parameter (element (0,0)) of the 2x2 matrices."""
        return self._get_parameter(0, 0)

    @Z11.setter
    def Z11(self, value: np.ndarray):
        """
        Zetter for the Z11 parameter.

        Args:
            value (numpy.ndarray): Values to set for the Z11 parameter.
        """
        self._set_parameter(0, 0, value)

    @property
    def Z12(self) -> np.ndarray:
        """Z12 parameter (element (0,1)) of the 2x2 matrices."""
        return self._get_parameter(0, 1)

    @Z12.setter
    def Z12(self, value: np.ndarray):
        """
        Zetter for the Z12 parameter.

        Args:
            value (numpy.ndarray): Values to set for the Z12 parameter.
        """
        self._set_parameter(0, 1, value)

    @property
    def Z21(self) -> np.ndarray:
        """Z21 parameter (element (1,0)) of the 2x2 matrices."""
        return self._get_parameter(1, 0)

    @Z21.setter
    def Z21(self, value: np.ndarray):
        """
        Zetter for the Z21 parameter.

        Args:
            value (numpy.ndarray): Values to set for the Z21 parameter.
        """
        self._set_parameter(1, 0, value)

    @property
    def Z22(self) -> np.ndarray:
        """Z22 parameter (element (1,1)) of the 2x2 matrices."""
        return self._get_parameter(1, 1)

    @Z22.setter
    def Z22(self, value: np.ndarray):
        """
        Zetter for the Z22 parameter.

        Args:
            value (numpy.ndarray): Values to set for the Z22 parameter.
        """
        self._set_parameter(1, 1, value)


def abcd_identity(N_abcd: int) -> ABCDArray:
    """
    Get abcd array of identity matrices.

    Args:
        N_abcd (int): Number of matrices in array.

    Returns:
        ABCDArray: Array of identity matrices.
    """
    return ABCDArray(np.array([[[1, 0], [0, 1]]] * N_abcd))
