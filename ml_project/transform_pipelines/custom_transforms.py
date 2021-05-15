import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class MedianScaler(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.median = 0.0
        self.factor = 1.0

    def transform(self, X, **kwargs):
        X -= self.median
        X /= self.factor
        return X

    def fit(self, X, y=None, **kwargs):
        self.median = X.apply(np.median)
        self.factor = (X - self.median).apply(np.abs).max()
        return self


class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, cols):
        self.cols = cols

    def fit(self, X: pd.DataFrame, y: pd.Series):
        return self

    def transform(self, X: pd.DataFrame):
        return X.drop(columns=self.cols)
