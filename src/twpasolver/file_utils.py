"""
Functions for saving to json and hdf5 files.

This module provides functions to read from and write to JSON and HDF5 files. It includes utility
functions to ensure file extensions and directories are correctly handled, as well as recursive
functions to handle nested dictionaries when working with HDF5 files.

"""

import json
import os
from typing import Any, Dict, List

import h5py
import numpy as np
from h5py import Group


def read_file(savename: str, writer: str = "json") -> Dict[str, Any]:
    """
    Read data from a file and return it as a dictionary.

    Args:
        savename (str): The name of the file to read.
        writer (str): The file format to use (default is "json").

    Returns:
        Dict[str, Any]: Dictionary containing the read data.
    """
    savename = add_extension(savename, writer)
    if not os.path.exists(savename):
        raise FileNotFoundError(f"The file '{savename}' does not exist.")
    return {"hdf5": read_hdf5, "json": read_json}.get(writer, read_hdf5)(savename)


def save_to_file(savename: str, d: Dict[str, Any], writer: str = "json") -> None:
    """
    Save data to a file in the specified format.

    Args:
        savename (str): The name of the file to save.
        d (Dict[str, Any]): Dictionary containing data to be saved.
        writer (str): The file format to use (default is "json").
    """
    savename = add_extension(savename, writer)
    ensure_directory_exists(savename)
    {"hdf5": save_to_hdf5, "json": save_to_json}.get(writer, save_to_hdf5)(savename, d)


def add_extension(filename: str, extension: str) -> str:
    """
    Add the specified extension to the filename if not already present.

    Args:
        filename (str): The name of the file.
        extension (str): The extension to add.

    Returns:
        str: Filename with the specified extension.
    """
    if not filename.endswith(f".{extension}"):
        return f"{filename}.{extension}"
    return filename


def ensure_directory_exists(filename: str) -> None:
    """
    Ensure that the directory for the given filename exists; create it if necessary.

    Args:
        filename (str): The name of the file.
    """
    directory = os.path.dirname(filename)
    if directory:
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)


def read_hdf5(savename: str) -> Dict[str, Any]:
    """
    Read data from an HDF5 file and return it as a dictionary.

    Args:
        savename (str): The name of the HDF5 file to read.

    Returns:
        Dict[str, Any]: Dictionary containing the read data.
    """
    with h5py.File(savename, "r") as f:
        return _recursively_load_dict_contents_from_group(f)


def read_json(savename: str) -> Dict[str, Any]:
    """
    Read data from a JSON file and return it as a dictionary.

    Args:
        savename (str): The name of the JSON file to read.

    Returns:
        Dict[str, Any]: Dictionary containing the read data.
    """
    with open(savename) as f:
        return json.load(f)


def save_to_hdf5(savename: str, d: Dict[str, Any]) -> None:
    """
    Save data to an HDF5 file.

    Args:
        savename (str): The name of the HDF5 file to save.
        d (Dict[str, Any]): Dictionary containing data to be saved.
    """
    if not isinstance(d, dict):
        return

    with h5py.File(savename, "w") as f:
        _recursively_save_dict_contents_to_group(f, d)


def save_to_json(savename: str, d: Dict[str, Any]) -> None:
    """
    Save data to a JSON file.

    Args:
        savename (str): The name of the JSON file to save.
        d (Dict[str, Any]): Dictionary containing data to be saved.
    """
    with open(savename, "w", encoding="utf-8") as fp:
        json.dump(d, fp, cls=NpEncoder, indent=4)


def _recursively_save_dict_contents_to_group(f: Group, d: Dict[str, Any]) -> None:
    """
    Recursively save dictionary contents to an HDF5 group.

    Args:
        f (Group): The HDF5 group.
        d (Dict[str, Any]): The dictionary to save.
    """
    for key, item in d.items():
        key = str(key)

        if isinstance(item, dict):
            subgroup = f.create_group(key)
            _recursively_save_dict_contents_to_group(subgroup, item)
        else:
            if isinstance(item, (list, tuple)):
                item = np.array(item)
            if isinstance(item, str):
                dtype = h5py.special_dtype(vlen=str)
            elif isinstance(item, np.ndarray):
                dtype = item.dtype
            else:
                dtype = type(item)
            f.create_dataset(key, data=item, dtype=dtype)


def _recursively_load_dict_contents_from_group(f: Group) -> Dict[str, Any]:
    """
    Recursively load dictionary contents from an HDF5 group.

    Args:
        f (Group): The HDF5 group.

    Returns:
        Dict[str, Any]: The loaded dictionary.
    """
    ans = {}
    for key, item in f.items():
        if isinstance(item, h5py.Dataset):
            ans[key] = (
                item[()].decode("utf-8") if isinstance(item[()], bytes) else item[()]
            )
        elif isinstance(item, Group):
            ans[key] = _recursively_load_dict_contents_from_group(f[key])
    return ans


class NpEncoder(json.JSONEncoder):
    """JSON encoder for handling NumPy types."""

    def default(self, o: Any) -> int | float | List[Any] | str:
        """
        Return default encoding for NumPy types.

        Args:
            o (Any): The object to encode.

        Returns:
            int | float | List[Any] | str: The encoded object.
        """
        if isinstance(o, np.integer):
            return int(o)
        elif isinstance(o, np.floating):
            return float(o)
        elif isinstance(o, np.ndarray):
            return o.tolist()
        elif isinstance(o, np.complexfloating):
            return str(o)
        elif isinstance(o, complex):
            return str(o)
        return json.JSONEncoder.default(self, o)
