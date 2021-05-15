import os

import pandas as pd
from py._path.local import LocalPath

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler

from src.run_pipeline import train, run_pipeline
from src.entities import (
    SplittingParams,
    TrainParams,
    PredictParams,
    ModelParams,
    TransformParams,
    ConfigParams,
)


def test_run_pipeline(
    tmpdir: LocalPath,
    dataset_path: str,
    target_col: str,
):

    params = ConfigParams(
        do_train=True,
        do_predict=True,
        bin_model_path=tmpdir.join("bin_model.json"),
        train_params=TrainParams(
            train_data_path=dataset_path,
            target_col=target_col,
            train_metric_path=tmpdir.join("metrics.json"),
            splitting_params=SplittingParams(val_size=0.2, random_state=239),
        ),
        transformer=TransformParams(
            builder="",
        ),
        model=ModelParams(
            params={},
            module="sklearn.tree.DecisionTreeClassifier",
        ),
        test_params=PredictParams(
            target_col=target_col,
            predict_metric_path=None,
            predict_data_path=dataset_path,
            out_predict_path=tmpdir.join("predicts.csv"),
        ),
    )

    run_pipeline(params)


def test_train(
    tmpdir: LocalPath,
    dataset_path: str,
    target_col: str,
):
    expected_metric_path = tmpdir.join("metrics.json")
    params = TrainParams(
        train_data_path=dataset_path,
        target_col=target_col,
        train_metric_path=expected_metric_path,
        splitting_params=SplittingParams(val_size=0.2, random_state=239),
    )
    pipeline = Pipeline(
        [("transformer", MinMaxScaler()), ("model", LogisticRegression())]
    )
    pipeline = train(pipeline, params)

    data = pd.read_csv(dataset_path)
    X = data.drop(columns=target_col)
    y = data[target_col]

    assert pipeline.score(X, y) > 0.8
    assert os.path.exists(params.train_metric_path)
