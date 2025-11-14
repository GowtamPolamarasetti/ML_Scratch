# my_ml_project/tests/test_naive_bayes.py

import numpy as np
import pytest
from numpy.testing import assert_array_equal, assert_allclose

try:
    from sklearn.datasets import load_iris
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from pyml.naive_bayes import GaussianNB, MultinomialNB
from pyml.metrics import accuracy_score

@pytest.mark.skipif(not SKLEARN_AVAILABLE, reason="scikit-learn not found")
def test_gaussian_nb_iris():
    X, y = load_iris(return_X_y=True)
    X = X[y != 2, :2]
    y = y[y != 2]
    gnb = GaussianNB()
    gnb.fit(X, y)
    y_pred = gnb.predict(X)
    assert gnb._is_fitted
    assert gnb._mean.shape == (2, 2)
    assert gnb._var.shape == (2, 2)
    assert gnb._priors.shape == (2,)
    assert_allclose(np.sum(gnb._priors), 1.0)
    assert accuracy_score(y, y_pred) > 0.98

def test_multinomial_nb_simple():
    X = np.array([[1, 1, 0], [1, 1, 0], [0, 1, 1], [0, 1, 1]])
    y = np.array([0, 0, 1, 1])
    mnb = MultinomialNB()
    mnb.fit(X, y)
    assert mnb._is_fitted
    assert mnb._class_log_prior.shape == (2,)
    assert mnb._feature_log_prob.shape == (2, 3)
    X_test = np.array([[1, 0, 0]])
    y_pred = mnb.predict(X_test)
    assert y_pred[0] == 0
    X_test_2 = np.array([[0, 0, 1]])
    y_pred_2 = mnb.predict(X_test_2)
    assert y_pred_2[0] == 1
    X_test_3 = np.array([[0, 1, 0]])
    y_pred_3 = mnb.predict(X_test_3)
    assert y_pred_3[0] == 0
