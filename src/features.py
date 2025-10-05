from __future__ import annotations
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class SelectColumns(BaseEstimator, TransformerMixin):
    def __init__(self, cols):
        self.cols = cols
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.cols]

class AmountLog(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X = X.copy()
        if "Amount" in X.columns:
            X["Amount_log1p"] = np.log1p(X["Amount"])
        return X

class TimeDerive(BaseEstimator, TransformerMixin):
    def __init__(self, time_col="event_time"):
        self.time_col = time_col
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X = X.copy()
        if self.time_col in X.columns:
            t = pd.to_datetime(X[self.time_col])
            X["hour"] = t.dt.hour
            X["dow"] = t.dt.dayofweek
        return X
