"""Generic container for arrays of TwoPortModels, including nested lists."""

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
        """Wrap nested lists of cells into ModelArrays, recursively."""
        return [cls(cells=c) if isinstance(c, list) else c for c in cells]

    def __getitem__(self, indices: slice | int) -> ModelArray | TwoPortModel:
        """Return the cell at an index, or a copy of the array with the sliced cells."""
        if isinstance(indices, slice):
            return self.model_copy(update={"cells": self.cells[indices]})
        return self.cells[indices]

    def append(self, other: TwoPortModel) -> None:
        """Append a model to the cells."""
        self.cells.append(other)
