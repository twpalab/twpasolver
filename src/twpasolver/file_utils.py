"""Read and write nested dictionaries as JSON or HDF5 files."""

import json
import os
from typing import Any, Dict, List

import h5py
import numpy as np
from h5py import Group


def read_file(savename: str | os.PathLike, writer: str = "json") -> Dict[str, Any]:
    """
    Read a dictionary from a file.

    Args:
        savename (str): File name; the extension is added if missing.
        writer (str): ``"json"`` or ``"hdf5"``.
    """
    savename = add_extension(savename, writer)
    if not os.path.exists(savename):
        raise FileNotFoundError(f"The file '{savename}' does not exist.")
    return {"hdf5": read_hdf5, "json": read_json}.get(writer, read_hdf5)(savename)


def save_to_file(
    savename: str | os.PathLike, d: Dict[str, Any], writer: str = "json"
) -> None:
    """
    Save a dictionary to a file, creating its directory if needed.

    Args:
        savename (str): File name; the extension is added if missing.
        d (Dict[str, Any]): Data to save.
        writer (str): ``"json"`` or ``"hdf5"``.
    """
    savename = add_extension(savename, writer)
    ensure_directory_exists(savename)
    {"hdf5": save_to_hdf5, "json": save_to_json}.get(writer, save_to_hdf5)(savename, d)


def add_extension(filename: str | os.PathLike, extension: str) -> str:
    """Add ``.extension`` to the filename if not already present."""
    filename = os.fspath(filename)
    if not filename.endswith(f".{extension}"):
        return f"{filename}.{extension}"
    return filename


def ensure_directory_exists(filename: str) -> None:
    """Create the parent directory of ``filename`` if it does not exist."""
    directory = os.path.dirname(filename)
    if directory:
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)


def read_hdf5(savename: str) -> Dict[str, Any]:
    """Read a dictionary from an HDF5 file."""
    with h5py.File(savename, "r") as f:
        return _recursively_load_dict_contents_from_group(f)


def read_json(savename: str) -> Dict[str, Any]:
    """Read a dictionary from a JSON file."""
    with open(savename) as f:
        return json.load(f)


def save_to_hdf5(savename: str, d: Dict[str, Any]) -> None:
    """Save a dictionary to an HDF5 file; a failed save keeps the previous file."""
    if not isinstance(d, dict):
        return

    # Write to a temporary file so that a failed save keeps the previous one.
    tmp = savename + ".tmp"
    try:
        with h5py.File(tmp, "w") as f:
            _recursively_save_dict_contents_to_group(f, d)
        os.replace(tmp, savename)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def save_to_json(savename: str, d: Dict[str, Any]) -> None:
    """Save a dictionary to a JSON file, encoding NumPy types."""
    with open(savename, "w", encoding="utf-8") as fp:
        json.dump(d, fp, cls=NpEncoder, indent=4)


def _recursively_save_dict_contents_to_group(f: Group, d: Dict[Any, Any]) -> None:
    """Recursively save a dictionary to an HDF5 group; ``None`` values are skipped."""
    if d and all(isinstance(key, (int, np.integer)) for key in d):
        f.attrs["int_keys"] = True
    for key, item in d.items():
        key = str(key)
        if item is None:  # e.g. no stopband in the grid; read back as missing
            continue

        if isinstance(item, (list, tuple)):
            array = None
            if not any(isinstance(i, dict) for i in item):
                try:
                    array = np.array(item)
                except ValueError:  # ragged
                    pass
            if array is None or array.dtype.kind in "OU":
                # Lists of dicts, strings or ragged items: one entry per index.
                subgroup = f.create_group(key)
                subgroup.attrs["list"] = True
                _recursively_save_dict_contents_to_group(
                    subgroup, dict(enumerate(item))
                )
                continue
            item = array

        if isinstance(item, dict):
            subgroup = f.create_group(key)
            _recursively_save_dict_contents_to_group(subgroup, item)
        else:
            if isinstance(item, str):
                dtype = h5py.special_dtype(vlen=str)
            elif isinstance(item, np.ndarray):
                dtype = item.dtype
            else:
                dtype = type(item)
            f.create_dataset(key, data=item, dtype=dtype)


def _recursively_load_dict_contents_from_group(f: Group) -> Dict[Any, Any]:
    """Recursively load a dictionary from an HDF5 group."""
    ans: Dict[Any, Any] = {}
    int_keys = bool(f.attrs.get("int_keys", False))
    for key, item in f.items():
        if isinstance(item, h5py.Dataset):
            value = (
                item[()].decode("utf-8") if isinstance(item[()], bytes) else item[()]
            )
        elif isinstance(item, Group):
            value = _recursively_load_dict_contents_from_group(item)
            if item.attrs.get("list", False):
                value = [value[i] for i in range(len(value))]
        ans[int(key) if int_keys else key] = value
    return ans


class NpEncoder(json.JSONEncoder):
    """JSON encoder for handling NumPy types."""

    def default(self, o: Any) -> int | float | List[Any] | str:
        """Encode NumPy scalars and arrays; complex numbers become strings."""
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, np.complexfloating):
            return str(o)
        if isinstance(o, complex):
            return str(o)
        return json.JSONEncoder.default(self, o)
