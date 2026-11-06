"""Test twpasolver.models.compose function."""

import pytest

from twpasolver.models.compose import compose
from twpasolver.models.oneport import OnePortArray
from twpasolver.models.twoportarrays import TwoPortArray


def test_compose_all_one_port_models(capacitor, inductor):
    """Test compose with OnePortModels sharing the parallel configuration."""
    result = compose(capacitor, inductor, parallel=False)
    assert isinstance(result, OnePortArray)
    assert result.parallel is False
    assert result.cells == [capacitor, inductor]


def test_compose_mixed_models(capacitor, twpa):
    """Test compose with mixed OnePortModel and TwoPortModel instances."""
    result = compose(capacitor, twpa)
    assert isinstance(result, TwoPortArray)
    assert result.cells == [capacitor, twpa]


def test_compose_different_parallel_configs(capacitor, inductor):
    """Test compose with OnePortModels of different parallel configurations."""
    inductor.twoport_parallel = True
    result = compose(capacitor, inductor)
    assert isinstance(result, TwoPortArray)
    assert result.cells == [capacitor, inductor]


def test_compose_no_models():
    """Test compose with no models."""
    with pytest.raises(RuntimeError):
        result = compose()
