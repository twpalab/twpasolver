import pytest


def test_mode_array_multichar_relation():
    """Multi-character mode names work in relations; unknown names raise."""
    from twpasolver.modes_rwa import Mode, ModeArray

    modes = [Mode("p2"), Mode("s"), Mode("c")]
    ma = ModeArray(modes, [["c", "p2-s"]], None)
    assert ma.symbolic_expressions["c"] == {"p2": 1.0, "s": -1.0}
    assert ma._propagate_frequencies_symbolic({"p2": 12.0, "s": 5.0})["c"] == 7.0
    with pytest.raises(KeyError):
        ModeArray(modes, [["c", "p3-s"]], None)
