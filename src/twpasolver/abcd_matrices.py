"""ABCD matrices array module."""

import numpy as np

from twpasolver.mathutils import matmul_2x2, matpow_2x2


class ABCDArray:
    """A class representing an array of ABCD matrices."""

    def __init__(self, mat: np.ndarray):
        """
        Initialize the ABCDArray instance.

        Parameters:
        - mat (numpy.ndarray): Input array of 2x2 matrices.
        - Z0 (float or int): Line impedance
        """
        mat = np.asarray(mat)
        if len(mat.shape) != 3 or mat.shape[-2:] != (2, 2):
            raise ValueError("Input must be array of 2x2 matrices.")
        self._abcd = mat

    def __repr__(self):
        """
        Return a string representation of the ABCDArray.

        Returns:
        - str: String representation of the ABCDArray.
        """
        return f"{self.__class__.__name__}({self._abcd})"

    def __array__(self):
        """Convert the ABCDArray to a numpy array."""
        return self._abcd

    def __matmul__(self, other: "ABCDArray") -> "ABCDArray":
        """
        Efficient matrix multiplication with another ABCDArray.

        Parameters:
        - other (ABCDArray): Another ABCDArray for matrix multiplication.

        Returns:
        - ABCDArray: Result of the matrix multiplication.
        """
        return self.__class__(matmul_2x2(self._abcd, other._abcd))

    def __pow__(self, exponent: int) -> "ABCDArray":
        """
        Efficient matrix exponentiation of the ABCDArray.

        Parameters:
        - exponent (int): The exponent to raise the ABCDArray to.

        Returns:
        - ABCDArray: Result of raising the ABCDArray to the specified power.
        """
        return self.__class__(matpow_2x2(self._abcd, exponent))

    def __getitem__(self, indices) -> "ABCDArray | np.ndarray":
        """Get value at indices."""
        if isinstance(indices, slice):
            return self.__class__(self._abcd[indices])
        return self._abcd[indices]

    def __setitem__(self, val: float | int, *indices):
        """Set value at indices."""
        self._abcd[indices] = val

    @property
    def shape(self):
        """Shape of the internal array."""
        return self._abcd.shape

    @property
    def len(self):
        """Length of the internal array."""
        return self._abcd.shape[0]

    def _get_parameter(self, i: int, k: int):
        """
        Get the specified parameter of the 2x2 matrices.

        Parameters:
        - i (int): Row index (0 or 1).
        - k (int): Column index (0 or 1).

        Returns:
        - numpy.ndarray: The specified parameter of the 2x2 matrices.
        """
        return self._abcd[:, i, k]

    def _set_parameter(self, i: int, k: int, values: np.ndarray):
        """
        Set the specified parameter of the 2x2 matrices.

        Parameters:
        - i (int): Row index (0 or 1).
        - k (int): Column index (0 or 1).
        - values (numpy.ndarray): Values to set for the specified parameter.
        """
        if values.ndim != 1 and len(values) != self.len:
            raise ValueError(f"Must provide a 1-D array of length {self.len}")
        self._abcd[:, i, k] = np.asarray(values)

    @property
    def A(self):
        """A parameter (element (0,0)) of the 2x2 matrices."""
        return self._get_parameter(0, 0)

    @A.setter
    def A(self, value: np.ndarray):
        """Setter for the A parameter."""
        self._set_parameter(0, 0, value)

    @property
    def B(self):
        """B parameter (element (0,1)) of the 2x2 matrices."""
        return self._get_parameter(0, 1)

    @B.setter
    def B(self, value: np.ndarray):
        """Setter for the B parameter."""
        self._set_parameter(0, 1, value)

    @property
    def C(self):
        """C parameter (element (1,0)) of the 2x2 matrices."""
        return self._get_parameter(1, 0)

    @C.setter
    def C(self, value: np.ndarray):
        """Setter for the C parameter."""
        self._set_parameter(1, 0, value)

    @property
    def D(self):
        """D parameter (element (1,1)) of the 2x2 matrices."""
        return self._get_parameter(1, 1)

    @D.setter
    def D(self, value: np.ndarray):
        """Setter for the D parameter."""
        self._set_parameter(1, 1, value)


def abcd_identity(N_abcd: int) -> ABCDArray:
    """Get abcd array of identity matrices."""
    return ABCDArray(np.array([[[1, 0], [0, 1]]] * N_abcd))
