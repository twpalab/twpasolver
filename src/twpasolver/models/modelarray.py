"""Generic container for arrays of TwoPortModels, including nested lists."""

from __future__ import annotations

from typing import Any, ClassVar

from pydantic import Field, field_validator

from twpasolver.twoport import TwoPortModel


class ModelArray(TwoPortModel):
    """
    Generic container for arrays of TwoPortModels.

    Cells are model instances (copied), dicts with the ``name`` of a model class
    (any ``TwoPortModel`` subclass that declares a ``name`` field, including user
    models defined after import) or nested lists, wrapped into arrays.
    """

    cell_base: ClassVar[type] = TwoPortModel
    cells: list[Any] = Field(
        default_factory=list,
        description="List of TwoPortModels representing the basic cells. Nested lists are allowed.",
    )

    @field_validator("cells", mode="before", check_fields=True)
    @classmethod
    def validate_nested_cells(cls, cells: list[Any]) -> list[TwoPortModel]:
        """Validate every cell by its model class; nested lists become arrays."""
        return [
            (
                cls(cells=c)
                if isinstance(c, list)
                else TwoPortModel.validate_cell(c, cls.cell_base)
            )
            for c in cells
        ]

    def __getitem__(self, indices: slice | int) -> ModelArray | TwoPortModel:
        """Return the cell at an index, or a copy of the array with the sliced cells."""
        if isinstance(indices, slice):
            return self.model_copy(update={"cells": self.cells[indices]})
        return self.cells[indices]

    def append(self, other: TwoPortModel) -> None:
        """Append a model to the cells."""
        self.cells = [*self.cells, other]
