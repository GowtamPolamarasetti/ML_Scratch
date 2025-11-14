# my_ml_project/tests/test_linear_model.py

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pyml.linear_model import LinearRegression, LogisticRegression
from pyml.preprocessing import StandardScaler
from pyml.metrics import r2_score, accuracy_score

@pytest.fixture
def linear_data():
    np.random.seed(42)
    X = np.random.rand(100, 2)
    y = 2 * X[:, 0] + 3 * X[:, 1] + 4.0 + np.random.randn(100) * 0.1
    X = StandardScaler().fit_transform(X)
    return X, y

def test_linear_regression(linear_data):
    X, y = linear_data
    reg = LinearRegression(learning_rate=0.1, n_iterations=1000, random_state=42)
    reg.fit(X, y)
    assert reg._is_fitted
    assert reg.coef_ is not None
    assert reg.intercept_ is not None
    y_pred = reg.predict(X)
    r2 = r2_score(y, y_pred)
    assert r2 > 0.95
    assert len(reg.history_['cost']) == 1000
    assert reg.history_['cost'][0] > reg.history_['cost'][-1]

@pytest.fixture
def logistic_data():
    np.random.seed(42)
    X = np.array([[0.1, 1.1], [0.2, 1.2], [0.3, 1.3],
                  [2.0, 3.0], [2.1, 3.1], [2.2, 3.2]])
    y = np.array([0, 0, 0, 1, 1, 1])
    X = StandardScaler().fit_transform(X)
    return X, y

def test_logistic_regression(logistic_data):
    X, y = logistic_data
    clf = LogisticRegression(learning_rate=0.1, n_iterations=1000, random_state=42)
    clf.fit(X, y)
    assert clf._is_fitted
    y_pred = clf.predict(X)
    acc = accuracy_score(y, y_pred)
    assert acc == 1.0
    probas = clf.predict_proba(X)
    assert probas.shape == (6, 2)
    assert_allclose(np.sum(probas, axis=1), 1.0)
    assert probas[0, 0] > 0.9
    assert probas[5, 1] > 0.9

def test_logistic_regression_multiclass_fail(logistic_data):
    X, y = logistic_data
    y_multi = np.array([0, 1, 2, 0, 1, 2])
    clf = LogisticRegression()
    with pytest.raises(ValueError, match="currently only supports binary"):
        clf.fit(X, y_multi)
