"""Generic models for user-defined abcd matrices and TwoPortModel lists."""

# mypy: ignore-errors
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
    def validate_nested_cells(cls, cells: list[TwoPortModel]):
        """Recursively create ModelArrays inside nested lists."""
        for i, c in enumerate(cells):
            if isinstance(c, list):
                cells[i] = cls(cells=c)
        return cells

    def __add__(self, other: ModelArray) -> ModelArray:
        """Operator to concatenate two ModelArrays."""
        return self.__class__(cells=[self, other])

    def __getitem__(self, indices: slice | int) -> ModelArray | TwoPortModel:
        """
        Get model(s) at indices or slice.

        Args:
            indices: Indices to access the internal AnyModel list.

        Returns:
            ModelArray | Anymodel: Value at the specified indices or slice.
        """
        if isinstance(indices, slice):
            return self.__class__(cells=self.cells[indices])
        return self.cells[indices]

    def append(self, other: TwoPortModel) -> None:
        """Append new model to internal cell array."""
        self.cells.append(other)
