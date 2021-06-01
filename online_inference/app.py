import logging
import os
import pickle

from typing import List, Optional

import pandas as pd
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, conint, confloat

from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


def load_object(path: str) -> Pipeline:
    with open(path, "rb") as f:
        return pickle.load(f)


class ModelInput(BaseModel):
    age: conint(ge=0, le=120)
    sex: conint(ge=0, le=1)
    cp: conint(ge=0, le=3)
    trestbps: conint(ge=0, le=300)
    chol: conint(ge=100, le=700)
    fbs: conint(ge=0, le=1)
    restecg: conint(ge=0, le=2)
    thalach: conint(ge=0, le=300)
    exang: conint(ge=0, le=1)
    oldpeak: confloat(ge=0, le=10.0)
    scope: conint(ge=0, le=2)
    ca: conint(ge=0, le=4)
    thal: conint(ge=0, le=3)

    class Config:
        extra = "forbid" #forbid option raise an exception when extra fields are determined


class ModelResponse(BaseModel):
    out: conint(ge=0, le=1)
    inp: ModelInput


model: Optional[Pipeline] = None


def make_predict(
    data: List[ModelInput],
    model: Pipeline,
) -> List[ModelResponse]:
    df = pd.DataFrame([d.dict() for d in data])
    predicts = model.predict(df)
    return [
        ModelResponse(inp=inp, out=pred)
        for pred, inp in zip(predicts, data)
        ]


app = FastAPI()


@app.get("/")
def main():
    return "it is entry point of health predictor"


@app.on_event("startup")
def load_model():
    global model
    model_path = os.getenv("PATH_TO_MODEL")
    if model_path is None:
        err = f"PATH_TO_MODEL {model_path} is None"
        logger.error(err)
        raise RuntimeError(err)

    model = load_object(model_path)


@app.get("/healz")
def health() -> bool:
    return not (model is None)


@app.post("/predict", response_model=List[ModelResponse])
def predict(request: List[ModelInput]):
    if request:
        return make_predict(request, model)
    else:
        return []


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=os.getenv("PORT", 8000))
