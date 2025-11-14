# my_ml_project/pyml/naive_bayes/_gaussian.py

import numpy as np
from ..base import BaseEstimator
from ..utils._validation import _check_X_y, _check_X

class GaussianNB(BaseEstimator):
    def __init__(self):
        self._classes = None
        self._priors = None
        self._mean = None
        self._var = None
        self._is_fitted = False
        self._n_features = None

    def fit(self, X, y):
        X, y = _check_X_y(X, y)
        n_samples, n_features = X.shape
        self._classes = np.unique(y)
        n_classes = len(self._classes)
        self._n_features = n_features
        self._mean = np.zeros((n_classes, n_features))
        self._var = np.zeros((n_classes, n_features))
        self._priors = np.zeros(n_classes)
        epsilon = 1e-9
        for idx, cls in enumerate(self._classes):
            X_cls = X[y == cls]
            self._mean[idx, :] = np.mean(X_cls, axis=0)
            self._var[idx, :] = np.var(X_cls, axis=0) + epsilon
            self._priors[idx] = X_cls.shape[0] / n_samples
        self._is_fitted = True
        return self

    def _gaussian_pdf(self, X, class_idx):
        mean = self._mean[class_idx]
        var = self._var[class_idx]
        numerator = np.exp(- (X - mean)**2 / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        log_pdf = np.sum(np.log(numerator / denominator), axis=1)
        return log_pdf

    def predict_log_proba(self, X):
        if not self._is_fitted:
            raise RuntimeError("GaussianNB must be fitted before predicting.")
        X = _check_X(X)
        if X.shape[1] != self._n_features:
            raise ValueError(f"Expected {self._n_features} features but got {X.shape[1]}")
        n_samples = X.shape[0]
        n_classes = len(self._classes)
        log_probas = np.zeros((n_samples, n_classes))
        for idx, cls in enumerate(self._classes):
            log_prior = np.log(self._priors[idx])
            log_likelihood = self._gaussian_pdf(X, idx)
            log_probas[:, idx] = log_likelihood + log_prior
        return log_probas

    def predict(self, X):
        log_probas = self.predict_log_proba(X)
        return self._classes[np.argmax(log_probas, axis=1)]
