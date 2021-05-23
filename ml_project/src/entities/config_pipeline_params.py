from dataclasses import dataclass
from .train_params import TrainParams, ModelParams, TransformParams
from .predict_params import PredictParams
from marshmallow_dataclass import class_schema
import yaml


@dataclass()
class ConfigParams:
    do_train: bool
    do_predict: bool
    bin_model_path: str
    train_params: TrainParams
    transformer: TransformParams
    model: ModelParams
    test_params: PredictParams


ConfigParamsSchema = class_schema(ConfigParams)


def read_config_params(path: str) -> ConfigParams:
    with open(path, "r") as input_stream:
        schema = ConfigParamsSchema()
        return schema.load(yaml.safe_load(input_stream))
