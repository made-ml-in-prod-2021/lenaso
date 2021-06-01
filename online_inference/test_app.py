from fastapi.testclient import TestClient
import pytest

from app import app, load_model

client = TestClient(app)


@pytest.fixture()
def default_values():
    values = {
       "age": 50,
       "sex": 1,
       "cp": 0,
       "trestbps": 120,
       "chol": 240,
       "fbs": 0,
       "restecg": 0,
       "thalach": 150,
       "exang": 0,
       "oldpeak": 0,
       "scope": 2,
       "ca": 0,
       "thal": 2
   }
    return values

def test_entry_point():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == "it is entry point of health predictor"


def test_model_healz():
    response = client.get("/healz")
    assert response.status_code == 200
    assert response.json() == False

    load_model()

    response = client.get("/healz")
    assert response.status_code == 200
    assert response.json() == True


def test_predict_without_body():
    with TestClient(app) as client:
        response = client.post("/predict")
        assert response.status_code == 422


def test_predict_correct_default(default_values):
    with TestClient(app) as client:
        response = client.post("/predict", json=[default_values])
        assert response.status_code == 200
        assert response.json() == [
            {
                "inp": default_values,
                "out": 1
            }
        ]


def test_predict_correct_multiple(default_values):
    with TestClient(app) as client:
        inp1 = default_values.copy()
        inp1["age"] = 40
        inp2 = default_values.copy()
        inp2["thal"] = 3

        response = client.post("/predict", json=[inp1, inp2])
        print(response.json())
        assert response.status_code == 200
        assert response.json() == [
            {
                "inp": inp1,
                "out": 1
            },
            {
                "inp": inp2,
                "out": 0
            },
        ]


def test_predict_missing_field(default_values):
    with TestClient(app) as client:
        inp1 = default_values.copy()
        del inp1["age"]

        response = client.post("/predict", json=[inp1])
        print(response.json())
        assert response.status_code == 422


def test_predict_extra_field(default_values):
    with TestClient(app) as client:
        inp1 = default_values.copy()
        inp1["age2"] = 30

        response = client.post("/predict", json=[inp1])
        assert response.status_code == 422


def test_predict_uncorrect_value(default_values):
    with TestClient(app) as client:
        inp1 = default_values.copy()
        inp1["sex"] = 2

        response = client.post("/predict", json=[inp1])
        assert response.status_code == 422