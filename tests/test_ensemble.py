# my_ml_project/tests/test_ensemble.py

import numpy as np
import pytest

try:
    from sklearn.datasets import load_iris, make_classification
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from pyml.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from pyml.metrics import accuracy_score

@pytest.mark.skipif(not SKLEARN_AVAILABLE, reason="scikit-learn not found")
def test_random_forest_iris():
    X, y = load_iris(return_X_y=True)
    clf = RandomForestClassifier(n_estimators=10, max_depth=5, random_state=42)
    clf.fit(X, y)
    y_pred = clf.predict(X)
    assert clf._is_fitted
    assert len(clf.trees_) == 10
    assert accuracy_score(y, y_pred) > 0.95

@pytest.mark.skipif(not SKLEARN_AVAILABLE, reason="scikit-learn not found")
def test_adaboost_simple():
    X, y = make_classification(n_samples=100, n_features=2, n_informative=2,
                               n_redundant=0, n_clusters_per_class=1, random_state=42)
    clf = AdaBoostClassifier(n_estimators=10, random_state=42)
    clf.fit(X, y)
    y_pred = clf.predict(X)
    assert clf._is_fitted
    assert len(clf.estimators_) > 0
    assert accuracy_score(y, y_pred) > 0.90

@pytest.mark.skipif(not SKLEARN_AVAILABLE, reason="scikit-learn not found")
def test_gradient_boosting_simple():
    X, y = make_classification(n_samples=100, n_features=2, n_informative=2,
                               n_redundant=0, n_clusters_per_class=1, random_state=42)
    clf = GradientBoostingClassifier(n_estimators=10, max_depth=3, learning_rate=0.1)
    clf.fit(X, y)
    y_pred = clf.predict(X)
    assert clf._is_fitted
    assert len(clf.trees_) == 10
    assert clf.initial_prediction_ is not None
    assert accuracy_score(y, y_pred) > 0.90
