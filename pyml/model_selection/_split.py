# my_ml_project/pyml/model_selection/_split.py

import numpy as np

def train_test_split(X, y, test_size=0.25, random_state=None):
    if X is None or y is None:
        raise ValueError("X and y must not be None.")
    X = np.asarray(X)
    y = np.asarray(y)
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples.")
    if random_state is not None:
        np.random.seed(random_state)
    n_samples = X.shape[0]
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    n_test = int(n_samples * test_size)
    n_train = n_samples - n_test
    train_indices = indices[:n_train]
    test_indices = indices[n_train:]
    X_train = X[train_indices]
    y_train = y[train_indices]
    X_test = X[test_indices]
    y_test = y[test_indices]
    return X_train, X_test, y_train, y_test
