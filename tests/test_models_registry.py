"""Cells validated by name, user models, nonlinear-element hook and model repetition."""

from typing import Literal

import numpy as np
import pytest
from pydantic import ValidationError

from twpasolver import TWPAnalysis
from twpasolver.cmes.bloch import series_nl_elements
from twpasolver.matrices_arrays import ABCDArray
from twpasolver.models import (
    PORT_IN,
    PORT_OUT,
    TWPA,
    Inductance,
    LCLfBaseCell,
    OnePortArray,
)
from twpasolver.twoport import TwoPortModel

CELL = dict(L=49.9e-12, C=24.2e-15, Lf=0.87e-9)


class UserSeriesL(TwoPortModel):
    """A user model defined after the package import."""

    name: Literal["UserSeriesL"] = "UserSeriesL"
    L: float

    def single_abcd(self, freqs):
        abcd = np.zeros((len(freqs), 2, 2), dtype=complex)
        abcd[:, 0, 0] = abcd[:, 1, 1] = 1
        abcd[:, 0, 1] = 2j * np.pi * np.asarray(freqs) * self.L
        return ABCDArray(abcd)


class HalfNonlinearCell(LCLfBaseCell):
    """Unnamed subclass with a custom nonlinear-element hook."""

    def bloch_nl_elements(self):
        return [(PORT_OUT, self.L / 2)]


def test_user_model_by_instance_dict_and_json():
    twpa = TWPA(cells=[UserSeriesL(L=1e-11), {"name": "UserSeriesL", "L": 2e-11}])
    assert [type(c) for c in twpa.cells] == [UserSeriesL, UserSeriesL]
    back = TWPA.model_validate(twpa.model_dump())
    assert back.cells[1].L == 2e-11 and isinstance(back.cells[1], UserSeriesL)


def test_unnamed_subclass_keeps_its_type_and_cells_are_copied():
    cell = HalfNonlinearCell(**CELL)
    twpa = TWPA(cells=[cell])
    assert type(twpa.cells[0]) is HalfNonlinearCell and twpa.cells[0] is not cell


def test_unknown_name_and_wrong_base_raise():
    with pytest.raises(ValidationError, match="Unknown model name"):
        TWPA(cells=[{"name": "NoSuchCell"}])
    with pytest.raises(ValidationError, match="OnePortModel"):
        OnePortArray(cells=[LCLfBaseCell(**CELL)])
    OnePortArray(cells=[Inductance(L=1e-9)])


def test_nl_scale_scales_nonlinear_elements():
    assert series_nl_elements(LCLfBaseCell(**CELL, nl_scale=0.5)) == [
        (PORT_OUT, CELL["L"] / 2)
    ]
    centered = LCLfBaseCell(**CELL, centered=True, nl_scale=0.5)
    assert series_nl_elements(centered) == [
        (PORT_IN, CELL["L"] / 4),
        (PORT_OUT, CELL["L"] / 4),
    ]


def test_nl_scale_matches_custom_hook_in_bloch_gain():
    kw = dict(N=20, Istar=6.3e-3, Idc=1e-3, Ip0=1e-3)
    grid = (0.05, 20, 1e-2)
    scaled = TWPAnalysis(
        twpa=TWPA(cells=[LCLfBaseCell(**CELL, nl_scale=0.5)], **kw), freqs=grid
    )
    hooked = TWPAnalysis(twpa=TWPA(cells=[HalfNonlinearCell(**CELL)], **kw), freqs=grid)
    sig = np.array([3.0, 5.0])
    g_scaled = scaled.gain(sig, pump=12.0, Is0=1e-8, model="bloch").gain_db
    g_hooked = hooked.gain(sig, pump=12.0, Is0=1e-8, model="bloch").gain_db
    np.testing.assert_array_equal(g_scaled, g_hooked)
    full = TWPAnalysis(twpa=TWPA(cells=[LCLfBaseCell(**CELL)], **kw), freqs=grid)
    assert np.all(full.gain(sig, pump=12.0, Is0=1e-8, model="bloch").gain_db > g_scaled)


def test_mul_copies_and_validates():
    cell = LCLfBaseCell(**CELL, N=2)
    tripled = cell * 3
    assert tripled.N == 6 and cell.N == 2 and tripled is not cell
    with pytest.raises(ValidationError):
        cell * -1


def test_unnamed_subclass_with_new_fields_raises():
    with pytest.raises(TypeError, match="declare name"):

        class Extra(LCLfBaseCell):  # pylint: disable=unused-variable
            extra: float = 1.0
