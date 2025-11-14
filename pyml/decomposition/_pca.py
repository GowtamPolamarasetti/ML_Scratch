# my_ml_project/pyml/decomposition/_pca.py

import numpy as np
from ..base import BaseEstimator, TransformerMixin
from ..utils._validation import _check_X

class PCA(BaseEstimator, TransformerMixin):
    def __init__(self, n_components):
        self.n_components = n_components
        self.components_ = None
        self.mean_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self._is_fitted = False

    def fit(self, X, y=None):
        X = _check_X(X)
        n_samples, n_features = X.shape
        if self.n_components > n_features:
            raise ValueError("n_components cannot be greater than n_features")
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_
        cov_matrix = np.cov(X_centered.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        self.components_ = eigenvectors[:, :self.n_components].T
        self.explained_variance_ = eigenvalues[:self.n_components]
        total_variance = np.sum(eigenvalues)
        self.explained_variance_ratio_ = self.explained_variance_ / total_variance
        self._is_fitted = True
        return self

    def transform(self, X):
        if not self._is_fitted:
            raise RuntimeError("PCA must be fitted before transforming.")
        X = _check_X(X)
        X_centered = X - self.mean_
        X_transformed = np.dot(X_centered, self.components_.T)
        return X_transformed
