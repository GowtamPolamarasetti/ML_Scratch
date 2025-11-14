# my_ml_project/pyml/preprocessing/_scalers.py

import numpy as np
from ..base import BaseEstimator, TransformerMixin
from ..utils._validation import _check_X

class StandardScaler(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.mean_ = None
        self.scale_ = None
        self._n_features = None
        self._is_fitted = False

    def fit(self, X, y=None):
        X = _check_X(X)
        self.mean_ = np.mean(X, axis=0)
        self.scale_ = np.std(X, axis=0)
        self._n_features = X.shape[1]
        self.scale_[self.scale_ == 0] = 1.0
        self._is_fitted = True
        return self

    def transform(self, X):
        if not self._is_fitted:
            raise RuntimeError("StandardScaler must be fitted before transforming.")
        X = _check_X(X)
        if X.shape[1] != self._n_features:
            raise ValueError(f"Expected {self._n_features} features but got {X.shape[1]}")
        return (X - self.mean_) / self.scale_

class MinMaxScaler(BaseEstimator, TransformerMixin):
    def __init__(self, feature_range=(0, 1)):
        if not (feature_range[0] < feature_range[1]):
            raise ValueError("Minimum of feature_range must be smaller than maximum.")
        self.min_ = feature_range[0]
        self.max_ = feature_range[1]
        self.data_min_ = None
        self.data_max_ = None
        self.data_range_ = None
        self.scale_ = None
        self._n_features = None
        self._is_fitted = False

    def fit(self, X, y=None):
        X = _check_X(X)
        self.data_min_ = np.min(X, axis=0)
        self.data_max_ = np.max(X, axis=0)
        self.data_range_ = self.data_max_ - self.data_min_
        self._n_features = X.shape[1]
        self.data_range_[self.data_range_ == 0] = 1.0
        self.scale_ = (self.max_ - self.min_) / self.data_range_
        self._is_fitted = True
        return self

    def transform(self, X):
        if not self._is_fitted:
            raise RuntimeError("MinMaxScaler must be fitted before transforming.")
        X = _check_X(X)
        if X.shape[1] != self._n_features:
            raise ValueError(f"Expected {self._n_features} features but got {X.shape[1]}")
        X_scaled = (X - self.data_min_) * self.scale_ + self.min_
        return X_scaled
