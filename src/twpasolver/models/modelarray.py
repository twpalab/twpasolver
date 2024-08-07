"""
Generic models for user-defined ABCD matrices and TwoPortModel lists.

This module provides a generic container class for arrays of TwoPortModels, allowing for the
organization and manipulation of multiple two-port network models, including nested lists.
"""

from __future__ import annotations

from typing import Any

from pydantic import Field, field_validator

from twpasolver.twoport import TwoPortModel


class ModelArray(TwoPortModel):
    """Generic container for arrays of TwoPortModels."""

    cells: list[Any] = Field(
        default_factory=list,
        description="List of TwoPortModels representing the basic cells. Nested lists are allowed.",
    )

    @field_validator("cells", mode="before", check_fields=True)
    @classmethod
    def validate_nested_cells(cls, cells: list[TwoPortModel]) -> list[TwoPortModel]:
        """
        Recursively create ModelArrays inside nested lists.

        Args:
            cells (list[TwoPortModel]): List of TwoPortModels or nested lists of TwoPortModels.

        Returns:
            list[TwoPortModel]: Validated and potentially nested ModelArrays.
        """
        for i, c in enumerate(cells):
            if isinstance(c, list):
                cells[i] = cls(cells=c)
        return cells

    def __getitem__(self, indices: slice | int) -> ModelArray | TwoPortModel:
        """
        Get model(s) at indices or slice.

        Args:
            indices (slice | int): Indices to access the internal cells list.

        Returns:
            ModelArray | TwoPortModel: Value at the specified indices or slice.
        """
        if isinstance(indices, slice):
            return self.__class__(cells=self.cells[indices])
        return self.cells[indices]

    def append(self, other: TwoPortModel) -> None:
        """
        Append new model to internal cell array.

        Args:
            other (TwoPortModel): The model to append to the cells list.
        """
        self.cells.append(other)
