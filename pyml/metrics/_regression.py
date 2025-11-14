# my_ml_project/pyml/metrics/_regression.py

import numpy as np

def mean_squared_error(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return np.mean((y_true - y_pred)**2)

def r2_score(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    tss = np.sum((y_true - np.mean(y_true))**2)
    if tss == 0:
        return 1.0 if np.allclose(y_true, y_pred) else 0.0
    rss = np.sum((y_true - y_pred)**2)
    return 1 - (rss / tss)
