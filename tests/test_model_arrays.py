"""Tests for twpasolve.models.model_array module."""

from twpasolver.models.model_arrays import ModelArray


def test_array_creation(capacitor, inductor):
    """Test array creation with simple list."""
    arr = ModelArray(cells=[capacitor, inductor])
    assert arr[0].model_dump() == capacitor.model_dump()
    assert arr[1].model_dump() == inductor.model_dump()


def test_array_creation_from_dict(capacitor, inductor):
    """Test array creation with simple list of model dictionaries."""
    arr = ModelArray(cells=[capacitor.model_dump(), inductor.model_dump()])
    assert arr[0].model_dump() == capacitor.model_dump()
    assert arr[1].model_dump() == inductor.model_dump()


def test_nested_array(capacitor, inductor):
    """Test array creation with nested list."""
    arr = ModelArray(cells=[capacitor, [inductor, capacitor]])
    assert arr[0].model_dump() == capacitor.model_dump()
    assert arr[1][0].model_dump() == inductor.model_dump()
    assert arr[1].model_dump() == ModelArray(cells=[inductor, capacitor]).model_dump()


def test_array_creation_from_file(tmpdir, capacitor, inductor, twpa):
    """Test array creation with nested list."""
    filename = str(tmpdir.join("array_model"))
    arr = ModelArray(cells=[capacitor, [inductor, capacitor], twpa])
    arr.dump_to_file(filename)
    arr_from_file = ModelArray.from_file(filename)
    assert arr.model_dump() == arr_from_file.model_dump()


def test_array_add(capacitor, inductor, twpa):
    """Test composition of arrays."""
    arr1 = ModelArray(cells=[capacitor, inductor])
    arr_sum = arr1 + twpa
    assert (
        arr_sum.model_dump()
        == ModelArray(cells=[[capacitor, inductor], twpa]).model_dump()
    )
    arr_sum2 = arr1 + arr1
    assert (
        arr_sum2.model_dump()
        == ModelArray(cells=[capacitor, inductor, capacitor, inductor]).model_dump()
    )
