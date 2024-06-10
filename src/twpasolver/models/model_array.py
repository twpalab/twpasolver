"""Generic models for user-defined abcd matrices and TwoPortModel lists."""

from __future__ import annotations

from typing import Union

import numpy as np
from pydantic import Field, field_validator

from twpasolver.matrices_arrays import ABCDArray, abcd_identity
from twpasolver.twoport import TwoPortModel


def all_subclasses(cls):
    """Recursively get all subclasses of a given class."""
    return set(cls.__subclasses__()).union(
        [s for c in cls.__subclasses__() for s in all_subclasses(c)]
    )


AnyModel = Union[tuple(all_subclasses(TwoPortModel))]


class ModelArray(TwoPortModel):
    """Generic container for arrays of TwoPortModels."""

    cells: list[AnyModel | ModelArray] = Field(
        description="List of TwoPortModels representing the basic cells. Nested lists are allowed."
    )

    @field_validator("cells", mode="before", check_fields=True)
    @classmethod
    def validate_nested_cells(cls, cells: list[AnyModel | ModelArray]):
        """Recursively create ModelArrays inside nested lists."""
        for i, c in enumerate(cells):
            if isinstance(c, list):
                cells[i] = cls(cells=c)
        return cells

    def single_abcd(self, freqs: np.ndarray) -> ABCDArray:
        """Compute abcd of combined models."""
        sc = abcd_identity(len(freqs))
        for cell in self.cells:
            sc = sc @ cell.get_abcd(freqs)
        return sc

    def __add__(self, other: ModelArray) -> ModelArray:
        """Operator to concatenate two ModelArrays."""
        if other.N == self.N:
            return self.__class__(
                cells=self.cells + other.cells, N=self.N, name=self.name
            )
        else:
            return self.__class__(cells=[self, other])
