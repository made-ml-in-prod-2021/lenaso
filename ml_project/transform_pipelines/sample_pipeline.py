import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

from .custom_transforms import MedianScaler, DropColumns


def build_pipeline() -> Pipeline:
    transformer = ColumnTransformer(
        [
            (
                "categorical_pipeline",
                Pipeline(
                    [
                        (
                            "impute",
                            SimpleImputer(
                                missing_values=np.nan, strategy="most_frequent"
                            ),
                        ),
                        ("ohe", OneHotEncoder()),
                    ]
                ),
                ["cp", "restecg", "slope"],
            ),
            (
                "custom_transform",
                MedianScaler(),
                ["oldpeak", "thalach"],
            ),
            (
                "min_max_scaler",
                MinMaxScaler(),
                ["age", "trestbps", "chol"],
            ),
        ],
        remainder="passthrough",
    )

    pipeline = Pipeline(
        steps=[
            ("transformer", transformer),
        ]
    )
    return pipeline
