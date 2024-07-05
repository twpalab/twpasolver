"""Tests for twpasolve.models.model_array module."""

from twpasolver.models.twoportarrays import TwoPortArray


def test_array_creation(capacitor, inductor):
    """Test array creation with simple list."""
    arr = TwoPortArray(cells=[capacitor, inductor])
    assert arr[0].model_dump() == capacitor.model_dump()
    assert arr[1].model_dump() == inductor.model_dump()


def test_array_creation_from_dict(capacitor, inductor):
    """Test array creation with simple list of model dictionaries."""
    arr = TwoPortArray(cells=[capacitor.model_dump(), inductor.model_dump()])
    assert arr[0].model_dump() == capacitor.model_dump()
    assert arr[1].model_dump() == inductor.model_dump()


def test_nested_array(capacitor, inductor):
    """Test array creation with nested list."""
    arr = TwoPortArray(cells=[capacitor, [inductor, capacitor]])
    assert arr[0].model_dump() == capacitor.model_dump()
    assert arr[1][0].model_dump() == inductor.model_dump()
    assert arr[1].model_dump() == TwoPortArray(cells=[inductor, capacitor]).model_dump()


def test_array_creation_from_file(tmpdir, capacitor, inductor, twpa):
    """Test array creation with nested list."""
    filename = str(tmpdir.join("array_model"))
    arr = TwoPortArray(cells=[capacitor, [inductor, capacitor], twpa])
    arr.dump_to_file(filename)
    arr_from_file = TwoPortArray.from_file(filename)
    assert arr.model_dump() == arr_from_file.model_dump()


def test_array_add(capacitor, inductor, twpa):
    """Test composition of arrays."""
    arr1 = TwoPortArray(cells=[capacitor, inductor])
    arr_sum = arr1 + twpa
    assert (
        arr_sum.model_dump()
        == TwoPortArray(cells=[[capacitor, inductor], twpa]).model_dump()
    )


def test_array_append(capacitor, inductor):
    """Test append function."""
    arr = TwoPortArray(cells=[capacitor])
    arr.append(inductor)
    assert arr.model_dump() == TwoPortArray(cells=[capacitor, inductor]).model_dump()
