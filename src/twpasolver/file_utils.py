"""Functions for saving to json and hdf5 files."""

import json
import os
from typing import Any, Dict, List

import h5py
import numpy as np
from h5py import Group
from h5py._hl.group import Group


def read_file(savename: str, writer: str = "json") -> Dict[str, Any]:
    """
    Read data from a file and return it as a dictionary.

    :param savename: The name of the file to read.
    :param writer: The file format to use (default is "json").
    :return: Dictionary containing the read data.
    """
    savename = add_extension(savename, writer)
    if not os.path.exists(savename):
        raise FileNotFoundError(f"The file '{savename}' does not exist.")
    return {"hdf5": read_hdf5, "json": read_json}.get(writer, read_hdf5)(savename)


def save_to_file(savename: str, d: Dict[str, Any], writer: str = "json") -> None:
    """
    Save data to a file in the specified format.

    :param savename: The name of the file to save.
    :param d: Dictionary containing data to be saved.
    :param writer: The file format to use (default is "json").
    :return: None
    """
    savename = add_extension(savename, writer)
    ensure_directory_exists(savename)
    {"hdf5": save_to_hdf5, "json": save_to_json}.get(writer, save_to_hdf5)(savename, d)


def add_extension(filename: str, extension: str) -> str:
    """Add the specified extension to the filename if not already present."""
    if not filename.endswith(f".{extension}"):
        return f"{filename}.{extension}"
    return filename


def ensure_directory_exists(filename: str) -> None:
    """Ensure that the directory for the given filename exists; create it if necessary."""
    directory = os.path.dirname(filename)
    if directory:
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)


def read_hdf5(savename: str) -> Dict[str, Any]:
    """Read data from an HDF5 file and return it as a dictionary."""
    with h5py.File(savename, "r") as f:
        return recursively_load_dict_contents_from_group(f)


def read_json(savename: str) -> Dict[str, Any]:
    """Read data from a JSON file and return it as a dictionary."""
    with open(savename) as f:
        return json.load(f)


def save_to_hdf5(savename: str, d: Dict[str, Any]) -> None:
    """Save data to an HDF5 file."""
    if not isinstance(d, dict):
        return

    with h5py.File(savename, "w") as f:
        recursively_save_dict_contents_to_group(f, d)


def save_to_json(savename: str, d: Dict[str, Any]) -> None:
    """Save data to a JSON file."""
    with open(savename, "w", encoding="utf-8") as fp:
        json.dump(d, fp, cls=NpEncoder, indent=4)


def recursively_save_dict_contents_to_group(f: Group, d: Dict[str, Any]) -> None:
    """Recursively save dictionary contents to an HDF5 group."""
    for key, item in d.items():
        key = str(key)

        if isinstance(item, dict):
            subgroup = f.create_group(key)
            recursively_save_dict_contents_to_group(subgroup, item)
        else:
            if isinstance(item, str):
                dtype = h5py.special_dtype(vlen=str)
            elif isinstance(item, list):
                dtype = h5py.special_dtype(vlen=list)
            elif isinstance(item, np.ndarray):
                dtype = item.dtype
            else:
                dtype = type(item)
            print(item, dtype)
            f.create_dataset(key, data=item, dtype=dtype)


def recursively_load_dict_contents_from_group(f: Group) -> Dict[str, Any]:
    """Recursively load dictionary contents from an HDF5 group."""
    ans = dict()
    for key, item in f.items():
        if isinstance(item, h5py.Dataset):
            ans[key] = (
                item[()].decode("utf-8") if isinstance(item[()], bytes) else item[()]
            )
        elif isinstance(item, Group):
            ans[key] = recursively_load_dict_contents_from_group(f[key])
    return ans


class NpEncoder(json.JSONEncoder):
    """JSON encoder for handling NumPy types."""

    def default(self, o: Any) -> int | float | List[Any] | str:
        """Return default encoding."""
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, np.complexfloating):
            return str(o)
        return json.JSONEncoder.default(self, o)
