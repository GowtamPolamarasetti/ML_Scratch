# my_ml_project/tests/test_svm.py

import numpy as np
import pytest

try:
    from sklearn.datasets import load_iris
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from pyml.svm import LinearSVC
from pyml.metrics import accuracy_score

@pytest.mark.skipif(not SKLEARN_AVAILABLE, reason="scikit-learn not found")
def test_linear_svc_iris():
    X, y = load_iris(return_X_y=True)
    X_bin = X[(y == 0) | (y == 1)]
    y_bin = y[(y == 0) | (y == 1)]
    X_bin = (X_bin - np.mean(X_bin, axis=0)) / np.std(X_bin, axis=0)
    svc = LinearSVC(learning_rate=0.01, n_iterations=1000, C=1.0)
    svc.fit(X_bin, y_bin)
    y_pred = svc.predict(X_bin)
    assert svc._is_fitted
    assert svc.coef_.shape == (X_bin.shape[1],)
    assert isinstance(svc.intercept_, float)
    assert accuracy_score(y_bin, y_pred) >= 1.0
