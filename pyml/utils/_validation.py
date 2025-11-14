# my_ml_project/pyml/utils/_validation.py

import numpy as np

def _check_X_y(X, y):
    if X is None or y is None:
        raise ValueError("X and y must not be None.")
    X = np.asarray(X)
    y = np.asarray(y)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if y.ndim != 1:
        raise ValueError(f"y must be a 1D array, but got {y.ndim} dimensions.")
    if X.shape[0] == 0:
        raise ValueError("X and y must have at least 1 sample, but got 0 samples.")
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"Inconsistent number of samples between X and y. Got {X.shape[0]} and {y.shape[0]}.")
    return X, y

def _check_X(X):
    if X is None:
        raise ValueError("X must not be None.")
    X = np.asarray(X)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if X.shape[0] == 0:
        raise ValueError("X must have at least 1 sample, but got 0 samples.")
    return X
