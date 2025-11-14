# my_ml_project/pyml/preprocessing/_encoders.py

import numpy as np
from ..base import BaseEstimator, TransformerMixin
from ..utils._validation import _check_X

class LabelEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.classes_ = None
        self._is_fitted = False

    def fit(self, y):
        y = np.asarray(y)
        if y.ndim != 1:
            raise ValueError("LabelEncoder.fit expects a 1D array.")
        self.classes_ = np.unique(y)
        self._is_fitted = True
        return self

    def transform(self, y):
        if not self._is_fitted:
            raise RuntimeError("LabelEncoder must be fitted before transforming.")
        y = np.asarray(y)
        if y.ndim != 1:
            raise ValueError("LabelEncoder.transform expects a 1D array.")
        class_to_int = {cls: i for i, cls in enumerate(self.classes_)}
        try:
            return np.array([class_to_int[label] for label in y])
        except KeyError as e:
            raise ValueError(f"y contains new labels not seen in fit: {e}")

    def inverse_transform(self, y):
        if not self._is_fitted:
            raise RuntimeError("LabelEncoder must be fitted before inverse_transforming.")
        y = np.asarray(y)
        if y.ndim != 1:
            raise ValueError("LabelEncoder.inverse_transform expects a 1D array.")
        if np.max(y) >= len(self.classes_) or np.min(y) < 0:
            raise ValueError("y contains invalid integers.")
        return self.classes_[y]

class OneHotEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, sparse_output=False):
        if sparse_output:
            raise NotImplementedError("Sparse output is not yet supported.")
        self.categories_ = None
        self._n_features = None
        self._is_fitted = False

    def fit(self, X, y=None):
        X = _check_X(X)
        self._n_features = X.shape[1]
        self.categories_ = []
        for i in range(self._n_features):
            col = X[:, i]
            max_val = int(np.max(col))
            self.categories_.append(np.arange(max_val + 1))
        self._is_fitted = True
        return self

    def transform(self, X):
        if not self._is_fitted:
            raise RuntimeError("OneHotEncoder must be fitted before transforming.")
        X = _check_X(X)
        if X.shape[1] != self._n_features:
            raise ValueError(f"Expected {self._n_features} features but got {X.shape[1]}")
        output_cols = sum(len(cats) for cats in self.categories_)
        n_samples = X.shape[0]
        X_out = np.zeros((n_samples, output_cols), dtype=int)
        current_col = 0
        for i in range(self._n_features):
            n_categories = len(self.categories_[i])
            for n in range(n_samples):
                category_val = int(X[n, i])
                if category_val < 0:
                    raise ValueError("OneHotEncoder cannot transform negative values.")
                if category_val >= n_categories:
                    pass
                else:
                    X_out[n, current_col + category_val] = 1
            current_col += n_categories
        return X_out
