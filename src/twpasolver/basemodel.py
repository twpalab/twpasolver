"""Custom overload of pydantic BaseModel."""

from __future__ import annotations

from pydantic import BaseModel as PydanticBaseModel

from twpasolver.file_utils import read_file, save_to_file


class BaseModel(PydanticBaseModel):
    """
    Overridden pydantic BaseModel.

    This class extends the pydantic BaseModel to exclude serializing optional arguments by default
    and adds functionality for dumping and loading from files.
    """

    def model_dump(
        self, exclude_none: bool = True, mode: str = "json", **kwargs
    ) -> dict:
        """
        Override the model_dump method to customize serialization.

        Args:
            exclude_none (bool): Whether to exclude fields that are None. Defaults to True.
            mode (str): The serialization mode. Defaults to "json".
            **kwargs: Additional keyword arguments for customization.

        Returns:
            dict: The serialized representation of the model.
        """
        return super().model_dump(exclude_none=exclude_none, mode=mode, **kwargs)

    @classmethod
    def from_file(cls, filename: str):
        """
        Load a model instance from a file.

        Args:
            filename (str): The path to the file from which to load the model.

        Returns:
            BaseModel: An instance of the model loaded from the file.
        """
        model_dict = read_file(filename, writer="json")
        return cls(**model_dict)

    def dump_to_file(self, filename: str) -> None:
        """
        Dump the model instance to a file.

        Args:
            filename (str): The path to the file where the model will be saved.
        """
        model_dict = self.model_dump()
        save_to_file(filename, model_dict, writer="json")
