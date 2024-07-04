"""Custom overload of pydantic BaseModel."""

from pydantic import BaseModel as PydanticBaseModel

from twpasolver.file_utils import read_file, save_to_file


class BaseModel(PydanticBaseModel):
    """
    Overriden pydantyc BaseModel.

    By default, excludes serializing optional arguments and adds dumping/loading from files.
    """

    def model_dump(self, exclude_none=True, **kwargs):
        """Override model_dump method."""
        return super().model_dump(exclude_none=exclude_none, **kwargs)

    @classmethod
    def from_file(cls, filename: str):
        """Load model from file."""
        model_dict = read_file(filename, writer="json")
        return cls(**model_dict)

    def dump_to_file(self, filename: str):
        """Dump model to file."""
        model_dict = self.model_dump()
        save_to_file(filename, model_dict, writer="json")
