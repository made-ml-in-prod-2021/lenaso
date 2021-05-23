import importlib
import json
import logging
import sys, os

from sklearn.pipeline import Pipeline
import hydra
from hydra import utils
from omegaconf import DictConfig, OmegaConf

from src.data import read_data, split_train_val_data, extract_target
from src.entities.config_pipeline_params import (
    ConfigParams,
    ConfigParamsSchema,
    TrainParams,
    PredictParams,
    # read_config_params,
)

from src.models import (
    serialize_model,
    load_model,
    evaluate_model,
    open_path,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler(sys.stdout)
logger.addHandler(handler)


def train(pipeline: Pipeline, train_params: TrainParams) -> Pipeline:
    logger.info(f"start train pipeline {pipeline} with params {train_params}")

    data = read_data(train_params.train_data_path)
    data.drop_duplicates()
    logger.info(f"data.shape is {data.shape}")

    train_df, val_df = split_train_val_data(data, train_params.splitting_params)
    logger.info(f"train_df.shape is {train_df.shape}")
    logger.info(f"val_df.shape is {val_df.shape}")

    X_train, y_train = extract_target(train_df, train_params.target_col)
    X_val, y_val = extract_target(val_df, train_params.target_col)

    pipeline.fit(X_train, y_train)

    predicts = pipeline.predict(X_val)
    metrics = evaluate_model(predicts, y_val)

    with open_path(train_params.train_metric_path, "w") as metric_file:
        json.dump(metrics, metric_file)
    logger.info(f"metrics is {metrics}")

    return pipeline


def predict(pipeline: Pipeline, test_params: PredictParams) -> str:
    logger.info(f"start test pipeline {pipeline} with params {test_params}")

    data = read_data(test_params.predict_data_path)
    logger.info(f"data.shape is {data.shape}")

    if test_params.target_col:
        data, target = extract_target(data, test_params.target_col)

    predicts = pipeline.predict(data)
    with open_path(test_params.out_predict_path, "w") as fout:
        fout.write("\n".join(predicts.astype(str)))
    logger.info(f"predicts write to {test_params.out_predict_path}")

    if test_params.target_col and test_params.predict_metric_path:
        metrics = evaluate_model(predicts, target)

        with open_path(test_params.predict_metric_path, "w") as metric_file:
            json.dump(metrics, metric_file)
        logger.info(f"metrics is {metrics}")

    return test_params.out_predict_path


def run_pipeline(params: ConfigParams):
    if params.do_train:
        pipeline = None
        if params.transformer.builder:
            try:
                pipeline = importlib.import_module(
                    params.transformer.builder
                ).build_pipeline()
            except Exception as e:
                logger.error(f"Can't import transformer: {e}")
                return
        try:
            module, model = params.model.module.rsplit(".", 1)
            module = importlib.import_module(module)
            model = getattr(module, model)(**params.model.params)
        except Exception as e:
            logger.error(f"Can't import model: {e}")
            return

        if pipeline:
            pipeline.steps.append(["model", model])
        else:
            pipeline = Pipeline([("model", model)])

        pipeline = train(pipeline, params.train_params)
        path_to_model = serialize_model(pipeline, params.bin_model_path)
        logger.info(f"model saved to {path_to_model}")
        if params.do_predict:
            predict(pipeline, params.test_params)
    elif params.do_predict:
        pipeline = load_model(params.bin_model_path)
        predict(pipeline, params.test_params)


@hydra.main(config_path="../configs/", config_name="config")
def my_app(cfg: DictConfig) -> None:
    logger.info(f"Working directory is {os.getcwd()}")
    logger.debug(OmegaConf.to_yaml(cfg))

    params = ConfigParamsSchema().load(cfg)
    # hydra change working directory, path to data need to be converted to absolute
    params.train_params.train_data_path = utils.to_absolute_path(
        params.train_params.train_data_path
    )
    params.test_params.predict_data_path = utils.to_absolute_path(
        params.test_params.predict_data_path
    )

    run_pipeline(params)
    logger.info("Done!")


if __name__ == "__main__":
    my_app()
