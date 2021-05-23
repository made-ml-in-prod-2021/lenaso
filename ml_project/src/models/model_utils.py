import pickle
from typing import Dict

import numpy as np
import pandas as pd
import os

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    average_precision_score,
    f1_score,
)
from sklearn.pipeline import Pipeline


def open_path(path, mode="w"):
    try:
        os.makedirs(os.path.dirname(path))
    except:
        pass
    return open(path, mode)


def evaluate_model(predicts: np.ndarray, target: pd.Series) -> Dict[str, float]:
    return {
        "accuracy": accuracy_score(target, predicts),
        "roc_auc": roc_auc_score(target, predicts),
        "ap": average_precision_score(target, predicts),
        "f1": f1_score(target, predicts),
    }


def serialize_model(model: Pipeline, output: str) -> str:
    with open_path(output, "wb") as f:
        pickle.dump(model, f)
    return output


def load_model(path: str) -> Pipeline:
    with open(path, "rb") as f:
        model = pickle.load(f)
    return model
