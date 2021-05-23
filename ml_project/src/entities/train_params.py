from dataclasses import dataclass, field
from typing import Any
from .split_params import SplittingParams


@dataclass()
class TrainParams:
    splitting_params: SplittingParams
    train_data_path: str = field(default="data/heart.csv")
    target_col: str = field(default="target")
    train_metric_path: str = field(default="models/train_metrics.json")


@dataclass()
class ModelParams:
    params: Any  # = field(default={"C": 1.0, "random_state": 3})
    module: str = field(default="sklearn.linear_model.LogisticRegression")


@dataclass()
class TransformParams:
    builder: str = field(default="")
