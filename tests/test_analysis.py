import numpy as np
import pytest
from pydantic import PrivateAttr

from twpasolver.analysis import (
    Analyzer,
    ExecutionRequest,
    TWPAnalysis,
    analysis_function,
)
from twpasolver.file_utils import read_file


@pytest.fixture
def execution_request_data():
    return {
        "name": "test_function",
        "kwargs": {"x": 2},
    }


@pytest.fixture
def analyzer_data(tmpdir, execution_request_data):
    return {
        "data_file": str(tmpdir.join("test_data.hdf5")),
        "run": [ExecutionRequest(**execution_request_data)],
    }


class SimpleAnalyzer(Analyzer):
    _allowed_functions = PrivateAttr(["test_function"])

    @analysis_function
    def test_function(self, x):
        return x

    def update_base_data(self):
        pass


@pytest.fixture
def analyzer_instance(analyzer_data):
    return SimpleAnalyzer(**analyzer_data)


@pytest.fixture
def twpanalysis_instance(twpa_model):
    return TWPAnalysis(twpa=twpa_model, freqs_arange=(0, 9, 1e-3))


def test_execution_request(execution_request_data):
    request = ExecutionRequest(**execution_request_data)
    assert request.name == "test_function"
    assert request.kwargs == {"x": 2}


def test_analysis_function(analyzer_instance):
    test_instance = analyzer_instance
    assert test_instance.test_function(2) == 2
    assert test_instance.data == {"test_function": 2}


def test_analyzer_save_data(analyzer_instance):
    writer = "hdf5"
    _ = analyzer_instance.test_function(2)
    analyzer_instance.save_data(writer=writer)
    data = read_file(analyzer_instance.data_file, writer=writer)
    assert data == {"test_function": 2}


def test_analyzer_dump_and_load_file(analyzer_data, tmpdir):
    dump_file = str(tmpdir.join("test_class.json"))
    analyzer = SimpleAnalyzer(**analyzer_data)
    analyzer.dump_to_file(dump_file)
    analyzer_from_file = SimpleAnalyzer.from_file(dump_file)
    assert analyzer == analyzer_from_file


def test_analyzer_execute(analyzer_data):
    executed_class = SimpleAnalyzer(**analyzer_data)
    assert executed_class.data == {"test_function": 2}
    analyzer_data["run"] = []
    not_executed_class = SimpleAnalyzer(**analyzer_data)
    assert not_executed_class.data == {}


def test_analyzer_parameter_sweep(analyzer_instance):
    x_sweep = np.arange(0, 10, 1)
    analyzer_instance.data = {}
    _ = analyzer_instance.parameter_sweep("test_function", "x", x_sweep)
    assert "test_function" not in analyzer_instance.data.keys()
    assert "test_function_sweep" in analyzer_instance.data.keys()
    assert all(
        np.array(list(analyzer_instance.data["test_function_sweep"]["x"].keys()))
        == x_sweep
    )
    assert all(
        np.array(list(analyzer_instance.data["test_function_sweep"]["x"].values()))
        == x_sweep
    )
