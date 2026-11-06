"""API and robustness tests for ``TWPAnalysis``, file I/O and the circuit models."""

import logging
import os
import warnings

import numpy as np
import pytest
from pydantic import ValidationError

from twpasolver.analysis import TWPAnalysis
from twpasolver.cmes.solver_config import BlochSolverConfig
from twpasolver.file_utils import read_file, save_to_file
from twpasolver.logger import log
from twpasolver.matrices_arrays import ABCDArray
from twpasolver.models import TWPA, LCLfBaseCell
from twpasolver.models.oneport import Capacitance, OnePortArray, Resistance
from twpasolver.modes_rwa import RWAAnalyzer

CELL_U = {"name": "LCLfBaseCell", "N": 30, "L": 49.9e-12, "C": 24.2e-15, "Lf": 0.87e-9}
CELL_L = {"name": "LCLfBaseCell", "N": 6, "L": 48.6e-12, "C": 8.2e-15, "Lf": 0.28e-9}
SIGNALS = np.array([3.0, 5.0, 6.0])
PUMP = 12.0


def _small_twpa(**kwargs):
    return TWPA(
        cells=[CELL_U, CELL_L, CELL_U], N=20, Istar=6.3e-3, Idc=1e-3, Ip0=1e-3, **kwargs
    )


@pytest.fixture(scope="module")
def analysis():
    an = TWPAnalysis(twpa=_small_twpa(), freqs=(0.05, 20, 1e-2))
    return an


def _bloch_gain(an, **kwargs):
    return an.gain(
        SIGNALS,
        pump=PUMP,
        Is0=1e-8,
        model="bloch",
        thin=20,
        bloch_config=BlochSolverConfig(passes=10, save_all_passes=True),
        **kwargs,
    )


def _assert_same(a, b):
    """Recursive equality of saved and reloaded data."""
    if isinstance(a, dict):
        assert set(a) == set(b)
        for key in a:
            _assert_same(a[key], b[key])
    elif isinstance(a, (list, tuple)) and any(isinstance(i, dict) for i in a):
        assert isinstance(b, list) and len(a) == len(b)
        for x, y in zip(a, b):
            _assert_same(x, y)
    elif isinstance(a, str):
        assert a == b
    else:
        a_arr = np.asarray(a)
        if a_arr.dtype.kind == "U":
            assert [str(x) for x in a] == [str(x) for x in b]
        else:
            np.testing.assert_array_equal(a_arr, np.asarray(b))


# TWPAnalysis end to end, HDF5 round trips


def test_twpanalysis_end_to_end(analysis, tmp_path):
    an = analysis
    simple = an.gain(SIGNALS, pump=PUMP, Is0=1e-8, model="simple", thin=20)
    assert np.all(np.isfinite(simple["gain_db"]))
    bloch = _bloch_gain(an)
    assert np.all(np.isfinite(bloch["gain_db"]))
    assert "gains_all_passes" in bloch
    pm = an.phase_matching(pump_arange=PUMP + np.arange(3), signal_arange=SIGNALS)
    assert pm["delta"].shape == (len(pm["signal_freqs"]), len(pm["pump_freqs"]))
    bw = an.bandwidth(bloch)
    assert bw["max_gain"] == np.max(bloch["gain_db"])

    name = str(tmp_path / "run.hdf5")
    results = {"gain": bloch, "phase_matching": pm, "bandwidth": bw}
    save_to_file(name, results, writer="hdf5")
    loaded = read_file(name, writer="hdf5")
    _assert_same(results, loaded)
    # Int dict keys come back as int, so plot_mode_currents keeps its labels.
    reverse = loaded["gain"]["mode_info"]["reverse_mode_map"]
    assert all(isinstance(k, int) for k in reverse)
    assert isinstance(loaded["gain"]["gains_all_passes"], list)


def test_sweep_of_gain_results_saves(analysis, tmp_path):
    res = [
        analysis.gain(SIGNALS[:1], pump=p, Is0=1e-8, model="simple", thin=20)
        for p in (11.0, 12.0)
    ]
    name = str(tmp_path / "sweep.hdf5")
    save_to_file(name, {"sweep": res}, writer="hdf5")
    _assert_same({"sweep": res}, read_file(name, writer="hdf5"))


def test_digit_keyed_dict_stays_dict(tmp_path):
    name = str(tmp_path / "d.hdf5")
    data = {"a": {"0": 1.0, "1": 2.0}, "b": [{"x": 1}, {"x": 2}], "c": {0: "p"}}
    save_to_file(name, data, writer="hdf5")
    assert read_file(name, writer="hdf5") == data


def test_failed_save_keeps_previous_file(tmp_path):
    name = str(tmp_path / "keep.hdf5")
    save_to_file(name, {"x": np.arange(3)}, writer="hdf5")
    with pytest.raises(Exception):
        save_to_file(name, {"bad": object()}, writer="hdf5")
    np.testing.assert_array_equal(read_file(name, writer="hdf5")["x"], np.arange(3))
    assert not os.path.exists(name + ".tmp")


# Base data state, k_star guard


def test_linear_follows_grid_and_unit():
    an = TWPAnalysis(twpa=_small_twpa(), freqs=(0.05, 20, 1e-2))
    n = len(an.linear.freqs)
    an.freqs = (0.05, 20, 2e-2)
    np.testing.assert_array_equal(an.linear.freqs, an.f / an.unit_multiplier)
    assert len(an.linear.freqs) < n
    an.unit = "MHz"
    np.testing.assert_array_equal(an.linear.freqs, an.f / an.unit_multiplier)
    an.freqs = np.array([1000.0, 2000.0])
    np.testing.assert_array_equal(an.linear.freqs, [1000.0, 2000.0])
    assert an.linear.k.shape == (1, 2)


def test_base_data_computed_on_demand():
    an = TWPAnalysis(twpa=_small_twpa(), freqs=(0.05, 20, 1e-2))
    g = an.gain(SIGNALS, pump=PUMP)["gain_db"]
    an.twpa.N = 40
    g_long = an.gain(SIGNALS, pump=PUMP)["gain_db"]
    assert np.all(np.isfinite(g)) and not np.allclose(g, g_long)


def test_k_star_guard_with_coarse_grid():
    an = TWPAnalysis(twpa=_small_twpa(), freqs=(3, 20, 1e-2))
    assert np.all(np.isfinite(an.linear.k_star))


# Stopband from the supercell half trace


def test_stopbands_do_not_depend_on_grid_end():
    bands = []
    for f_max in (10, 20, 40):
        an = TWPAnalysis(twpa=_small_twpa(), freqs=(0.05, f_max, 1e-2))
        bands.append(an.linear.stopbands)
    # every band is listed; only the one cut by the grid end changes
    assert bands[2].shape[1] == 2 and len(bands[2]) > 3
    assert np.all(bands[2][:, 0] < bands[2][:, 1])
    assert np.all(bands[2][:-1, 1] < bands[2][1:, 0])
    np.testing.assert_array_equal(bands[0], bands[2][:1])
    np.testing.assert_array_equal(bands[1][:2], bands[2][:2])


def _flat_line_analysis():
    """LC ladder with a ~300 GHz cutoff: dispersionless and without stopband."""
    twpa = TWPA(
        cells=[LCLfBaseCell(L=50e-12, C=20e-15, Lf=0.0)],
        N=2000,
        Istar=6.3e-3,
        Idc=1e-3,
        Ip0=1e-3,
    )
    an = TWPAnalysis(twpa=twpa, freqs=(0.05, 30, 1e-2))
    return an


def test_no_stopband_gives_empty_stopbands():
    an = _flat_line_analysis()
    assert an.linear.stopbands.shape == (0, 2)
    assert np.all(np.isfinite(an.linear.k_star))


def test_gain_needs_pump(analysis):
    with pytest.raises(TypeError):
        analysis.gain(SIGNALS)


# 4WM Kerr mismatch


@pytest.mark.parametrize(
    "mixing, config, pump, kerr",
    [("4WM", "basic_4wm", 7.0, -2), ("3WM", "basic_3wm", 14.0, -1)],
)
def test_phase_matching_kerr_mismatch_on_flat_line(mixing, config, pump, kerr):
    """Kerr part of dk on a flat line: -2 chi k_p (4WM), -chi k_p (3WM), as in the CMEs."""
    an = _flat_line_analysis()
    Ip = 1e-3
    kw = dict(
        mixing=mixing,
        mode_array_config=config,
        pump_arange=[pump],
        signal_arange=[5.0, 6.0],
        relative=False,
    )
    dk = an.phase_matching(pump_Ip0=Ip, **kw)
    dk_lin = an.phase_matching(pump_Ip0=0.0, **kw)["delta"]
    chi = an.twpa.xi * Ip**2 / 8
    k_p = dk["triplets"]["k"][0][0]
    np.testing.assert_allclose((dk["delta"] - dk_lin) / (chi * k_p), kerr, rtol=1e-3)


# gain(Ip0=...) is local, position grid


def test_gain_ip0_does_not_change_twpa(analysis):
    ref = TWPAnalysis(
        twpa=_small_twpa().model_copy(update={"Ip0": 5e-4}), freqs=(0.05, 20, 1e-2)
    )
    for model in ("simple", "bloch"):
        kw = dict(pump=PUMP, Is0=1e-8, model=model, thin=20)
        if model == "bloch":
            kw["bloch_config"] = BlochSolverConfig(passes=10)
        res = analysis.gain(SIGNALS, Ip0=5e-4, **kw)
        assert analysis.twpa.Ip0 == 1e-3
        np.testing.assert_array_equal(
            res["gain_db"], ref.gain(SIGNALS, **kw)["gain_db"]
        )


def test_gain_position_grid_has_two_points_at_least(analysis):
    res = analysis.gain(
        SIGNALS[:1],
        pump=PUMP,
        model="simple",
        thin=10 * analysis.twpa.N_tot,
    )
    assert len(res["x"]) == 2


# phase_matching ranges


def test_phase_matching_explicit_ranges(analysis):
    pm = analysis.phase_matching(pump_arange=[13, 14, 15], signal_arange=[3.0, 5.0])
    np.testing.assert_array_equal(pm["pump_freqs"], [13, 14, 15])
    pm = analysis.phase_matching(pump_arange=(13, 14, 0.1), signal_arange=[3.0])
    assert len(pm["pump_freqs"]) == 10


def test_phase_matching_segmented_is_length_weighted_mean():
    segs = [
        TWPA(cells=[CELL_U, CELL_L, CELL_U], N=20, Istar=6.3e-3, Idc=1e-3, Ip0=1e-3),
        TWPA(cells=[CELL_U, CELL_L, CELL_U], N=10, Istar=9e-3, Idc=1e-3, Ip0=1e-3),
    ]
    kw = dict(pump_arange=[PUMP], signal_arange=SIGNALS, relative=False)
    f = (0.05, 20, 1e-2)
    dev = TWPA(cells=segs, Ip0=1e-3)
    delta = TWPAnalysis(twpa=dev, freqs=f).phase_matching(**kw)["delta"]
    parts = [TWPAnalysis(twpa=s, freqs=f).phase_matching(**kw)["delta"] for s in segs]
    w = np.array([s.N_tot for s in segs], dtype=float)
    assert np.all(np.isfinite(delta))
    np.testing.assert_allclose(delta, (w[0] * parts[0] + w[1] * parts[1]) / w.sum())


def test_phase_matching_zero_pump_is_linear(analysis):
    kw = dict(pump_arange=[13.0], signal_arange=[3.0, 5.0], relative=False)
    lin = analysis.phase_matching(pump_Ip0=0.0, **kw)["delta"]
    kerr = analysis.phase_matching(**kw)["delta"]
    pm = analysis.phase_matching(pump_Ip0=0.0, **kw)
    k_p, k_s, k_i = np.array(pm["triplets"]["k"]).T
    np.testing.assert_allclose(lin[:, 0], k_p - k_s - k_i)
    assert not np.allclose(lin, kerr)


# bloch with 2-D initial amplitudes


def test_bloch_2d_initial_amplitudes(analysis):
    labels = list(analysis.get_mode_array("basic_3wm").modes)
    y0 = np.zeros(len(labels), dtype=complex)
    y0[labels.index("p")] = analysis.twpa.Ip0
    y0[labels.index("s")] = 1e-8
    ref = _bloch_gain(analysis, initial_amplitudes_fwd=y0)
    res = _bloch_gain(analysis, initial_amplitudes_fwd=np.tile(y0, (len(SIGNALS), 1)))
    np.testing.assert_array_equal(res["gain_db"], ref["gain_db"])


# bandwidth


def test_bandwidth_of_nan_gain_raises():
    nan_gain = {
        "gain_db": np.full(3, np.nan),
        "signal_freqs": SIGNALS,
        "pump_freq": PUMP,
    }
    with pytest.raises(ValueError, match="NaN"):
        TWPAnalysis.bandwidth(nan_gain)


# Logging


def test_log_does_not_reach_root_handler():
    records: list = []
    handler = logging.Handler()
    handler.emit = records.append  # type: ignore[method-assign]
    logging.getLogger().addHandler(handler)
    try:
        log.warning("printed once, by the twpasolver handler")
    finally:
        logging.getLogger().removeHandler(handler)
    assert records == [] and log.handlers


# Validation


def test_twpanalysis_rejects_unknown_fields():
    with pytest.raises(ValidationError):
        TWPAnalysis(twpa=_small_twpa(), freqs=(0, 9, 1e-3), freqs_arange=(0, 9, 1e-3))


def test_port_must_be_a_model():
    with pytest.raises(ValidationError, match="TwoPortModel"):
        TWPAnalysis(twpa=_small_twpa(), freqs=(1, 2, 0.5), port=[50.0, 50.0])


def test_twpanalysis_json_round_trip_keeps_grid(tmp_path):
    an = TWPAnalysis(twpa=_small_twpa(), freqs=(0.05, 20, 1e-2), unit="GHz")
    path = str(tmp_path / "analysis.json")
    an.dump_to_file(path)
    np.testing.assert_array_equal(TWPAnalysis.from_file(path).f, an.f)


def test_warmup_runs():
    from twpasolver.cmes.warmup import warmup

    warmup(verbose=False)


def test_frequencies_validation():
    with pytest.raises(ValidationError):
        TWPAnalysis(twpa=_small_twpa(), freqs=(1, 2, 0))
    with pytest.raises(ValidationError, match="empty"):
        TWPAnalysis(twpa=_small_twpa(), freqs=(2, 1, 0.5))
    with pytest.raises(ValidationError):
        TWPAnalysis(twpa=_small_twpa())


def test_impedance_must_have_positive_real_part():
    with pytest.raises(ValidationError, match="must be positive"):
        TWPA(cells=[CELL_U], Z0_ref=0)


def test_abcd_setter_rejects_wrong_length():
    abcd = ABCDArray(np.tile(np.eye(2, dtype=complex), (2, 1, 1)))
    with pytest.raises(ValueError):
        abcd.A = np.array([5])
    abcd.A = [3, 4]
    np.testing.assert_array_equal(abcd.A, [3, 4])


# Models


def test_twpa_slice_keeps_parameters():
    twpa = TWPA(cells=[CELL_U, CELL_L], Istar=1e-3, Idc=2e-4, N=7)
    part = twpa[0:1]
    assert (part.Istar, part.Idc, part.N) == (1e-3, 2e-4, 7)
    assert len(part.cells) == 1
    arr = OnePortArray(
        parallel=True, cells=[Resistance(R=1), Capacitance(C=1e-12), Resistance(R=2)]
    )
    assert arr[0:2].parallel


def test_nested_cells_do_not_mutate_input():
    cells = [Resistance(R=1), [Resistance(R=2), Resistance(R=3)]]
    OnePortArray(cells=cells)
    assert isinstance(cells[1], list)


def test_abcd_array_protocol():
    cell = LCLfBaseCell(L=49.9e-12, C=24.2e-15, Lf=0.87e-9).get_cell(
        np.array([1e9, 2e9])
    )
    assert np.asarray(cell.abcd, dtype=complex).dtype == complex
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        np.array(cell.abcd)


def test_network_s_matches_for_complex_z0():
    model = LCLfBaseCell(L=49.9e-12, C=24.2e-15, Lf=0.87e-9, Z0_ref=30 + 20j)
    cell = model.get_cell(np.array([1e9, 5e9]))
    np.testing.assert_allclose(cell.to_network().s, np.asarray(cell.s), atol=1e-12)


def test_zero_repetitions_is_identity():
    freqs = np.array([1e9, 2e9])
    abcd = (LCLfBaseCell(L=49.9e-12, C=24.2e-15, Lf=0.87e-9) * 0).get_abcd(freqs)
    np.testing.assert_array_equal(np.asarray(abcd), np.tile(np.eye(2), (2, 1, 1)))


def test_resistance_on_integer_freqs():
    np.testing.assert_array_equal(Resistance(R=2.5).Z(np.array([1, 2])), [2.5, 2.5])


# modes_rwa


def test_cyclic_relations_raise():
    with pytest.raises(ValueError, match="cyclic"):
        RWAAnalyzer(["a", "b", "c"], [["a", "b+c"], ["b", "a-c"]])
