"""Custom overload of pydantic BaseModel."""

from __future__ import annotations

import os

from pydantic import BaseModel as PydanticBaseModel

from twpasolver.file_utils import read_file, save_to_file


class BaseModel(PydanticBaseModel):
    """Pydantic BaseModel that skips None fields when serializing and adds file I/O."""

    def model_dump(
        self, exclude_none: bool = True, mode: str = "json", **kwargs
    ) -> dict:
        """Serialize the model, by default in JSON mode and without None fields."""
        return super().model_dump(exclude_none=exclude_none, mode=mode, **kwargs)

    @classmethod
    def from_file(cls, filename: str | os.PathLike):
        """Load a model instance from a JSON file."""
        model_dict = read_file(filename, writer="json")
        return cls(**model_dict)

    def dump_to_file(self, filename: str | os.PathLike) -> None:
        """Dump the model instance to a JSON file."""
        model_dict = self.model_dump()
        save_to_file(filename, model_dict, writer="json")
