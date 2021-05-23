import os

from py._path.local import LocalPath

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from src.models.model_utils import open_path, serialize_model, load_model


def test_open_path(tmpdir: LocalPath):
    filepath = tmpdir.join("dir/file.txt")
    with open_path(filepath, "w") as fout:
        fout.write("test")
    with open(filepath, "r") as fin:
        text = fin.read()
    assert text == "test", f"text is {text} but expected: test"


def test_serialize_load(tmpdir: LocalPath):
    expected_output = tmpdir.join("model.pkl")
    model = Pipeline([("model", LogisticRegression())])
    real_output = serialize_model(model, expected_output)
    assert real_output == expected_output
    assert os.path.exists
    model2 = load_model(expected_output)
    assert isinstance(model2, Pipeline)
