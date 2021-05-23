from dataclasses import dataclass, field
from typing import Optional


@dataclass()
class PredictParams:
    target_col: Optional[str]
    predict_metric_path: Optional[str]
    predict_data_path: str = field(default="data/heart.csv")
    out_predict_path: str = field(default="models/test_predict.csv")
