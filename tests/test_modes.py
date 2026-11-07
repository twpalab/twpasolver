"""ModeSet: frequencies, RWA terms and relation checks."""

import json
from pathlib import Path

import numpy as np
import pytest

from twpasolver.modes_rwa import ModeSet

# Term lists of the former sympy implementation (5.3), frozen as reference.
RWA_REFERENCE = Path(__file__).parent / "data" / "rwa_terms.json"


@pytest.mark.parametrize("case", json.loads(RWA_REFERENCE.read_text()))
def test_rwa_terms_match_reference(case):
    modes = ModeSet(case["labels"], case["relations"])
    for power, ref in case["terms"].items():
        got = [[t[0], list(t[1]), t[4]] for t in modes.rwa_terms(int(power))]
        assert got == ref


def test_frequencies_over_several_independent_tones():
    modes = ModeSet(["p1", "p2", "s", "i"], [["i", "p1+p2-s"]])
    f = modes.frequencies({"p1": 7.0, "p2": 8.0, "s": np.array([5.0, 6.0])})
    np.testing.assert_array_equal(f[modes.index("i")], [10.0, 9.0])
    assert f.shape == (4, 2)
    with pytest.raises(ValueError, match="missing"):
        modes.frequencies({"p1": 7.0, "s": 5.0})


def test_relation_checks():
    with pytest.raises(KeyError):
        ModeSet(["p2", "s", "c"], [["c", "p3-s"]])
    with pytest.raises(ValueError, match="cyclic"):
        ModeSet(["a", "b"], [["a", "b"], ["b", "a"]])
    with pytest.raises(ValueError, match="more than one"):
        ModeSet(["p", "s", "i"], [["i", "p-s"], ["i", "p+s"]])


def test_expressions_and_dependents():
    modes = ModeSet.extended_3wm(n_pump_harmonics=1, n_frequency_conversion=1)
    assert modes.expression("i") == "p - s"
    assert modes.expression("p2") == "2*p"
    assert modes.dependents("s") == {"i", "ps", "pi", "s2", "i2"}
    assert modes.independent == ("p", "s")
