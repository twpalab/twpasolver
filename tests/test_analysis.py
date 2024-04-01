import pytest
from pydantic import PrivateAttr

from twpasolver.analysis import Analyzer, ExecutionRequest, analysis_function
from twpasolver.file_utils import read_file


@pytest.fixture
def execution_request_data():
    return {
        "name": "test_function",
        "parameters": {"x": 2},
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
def test_analyzer_class(analyzer_data):
    return SimpleAnalyzer(**analyzer_data)


def test_execution_request(execution_request_data):
    request = ExecutionRequest(**execution_request_data)
    assert request.name == "test_function"
    assert request.parameters == {"x": 2}


def test_analysis_function(test_analyzer_class):
    test_instance = test_analyzer_class
    assert test_instance.test_function(2) == 2
    assert test_instance.data == {"test_function": 2}


def test_analyzer_save_data(test_analyzer_class):
    writer = "hdf5"
    _ = test_analyzer_class.test_function(2)
    test_analyzer_class.save_data(writer=writer)
    data = read_file(test_analyzer_class.data_file, writer=writer)
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
