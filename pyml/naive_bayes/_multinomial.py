# my_ml_project/pyml/naive_bayes/_multinomial.py

import numpy as np
from ..base import BaseEstimator
from ..utils._validation import _check_X_y, _check_X

class MultinomialNB(BaseEstimator):
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self._classes = None
        self._class_log_prior = None
        self._feature_log_prob = None
        self._is_fitted = False
        self._n_features = None

    def fit(self, X, y):
        X, y = _check_X_y(X, y)
        if np.min(X) < 0:
            raise ValueError("MultinomialNB requires non-negative feature values.")
        n_samples, n_features = X.shape
        self._classes = np.unique(y)
        n_classes = len(self._classes)
        self._n_features = n_features
        class_counts = np.zeros(n_classes)
        for idx, cls in enumerate(self._classes):
            class_counts[idx] = np.sum(y == cls)
        self._class_log_prior = np.log(class_counts / n_samples)
        feature_counts_per_class = np.zeros((n_classes, n_features))
        for idx, cls in enumerate(self._classes):
            X_cls = X[y == cls]
            feature_counts_per_class[idx, :] = np.sum(X_cls, axis=0)
        total_counts_per_class = np.sum(feature_counts_per_class, axis=1)
        numerator = feature_counts_per_class + self.alpha
        denominator = total_counts_per_class.reshape(-1, 1) + self.alpha * n_features
        self._feature_log_prob = np.log(numerator / denominator)
        self._is_fitted = True
        return self

    def predict_log_proba(self, X):
        if not self._is_fitted:
            raise RuntimeError("MultinomialNB must be fitted before predicting.")
        X = _check_X(X)
        if X.shape[1] != self._n_features:
            raise ValueError(f"Expected {self._n_features} features but got {X.shape[1]}")
        jll = np.dot(X, self._feature_log_prob.T)
        jll += self._class_log_prior
        return jll

    def predict(self, X):
        log_probas = self.predict_log_proba(X)
        return self._classes[np.argmax(log_probas, axis=1)]
