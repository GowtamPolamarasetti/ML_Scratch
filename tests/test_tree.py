# my_ml_project/tests/test_tree.py
import numpy as np
import pytest
from numpy.testing import assert_array_equal

from pyml.tree import DecisionTreeClassifier, DecisionTreeRegressor
from pyml.metrics import accuracy_score, r2_score

@pytest.fixture
def simple_tree_data():
    X = np.array([[0.5], [0.8], [1.1], [1.5]])
    y = np.array([0, 0, 1, 1])
    return X, y

def test_decision_tree_classifier_simple(simple_tree_data):
    X, y = simple_tree_data
    clf = DecisionTreeClassifier(max_depth=1)
    clf.fit(X, y)
    assert clf._is_fitted
    assert clf.root is not None
    assert clf.root.feature_index == 0
    assert clf.root.threshold == 0.8
    y_pred = clf.predict(X)
    assert accuracy_score(y, y_pred) == 1.0
    assert clf.predict([[0.7]])[0] == 0
    assert clf.predict([[0.8]])[0] == 0
    assert clf.predict([[0.81]])[0] == 1
    assert clf.predict([[1.3]])[0] == 1

def test_decision_tree_classifier_depth(simple_tree_data):
    X, y = simple_tree_data
    clf = DecisionTreeClassifier(max_depth=0)
    clf.fit(X, y)
    assert clf.root.is_leaf_node()
    assert clf.root.value == 0
    y_pred = clf.predict(X)
    assert_array_equal(y_pred, [0, 0, 0, 0])

def test_decision_tree_classifier_min_samples(simple_tree_data):
    X, y = simple_tree_data
    clf = DecisionTreeClassifier(min_samples_split=5)
    clf.fit(X, y)
    assert clf.root.is_leaf_node()

@pytest.fixture
def simple_reg_data():
    X = np.array([[1], [2], [3], [4], [5]])
    y = np.array([2.1, 3.9, 6.2, 7.8, 10.1])
    return X, y

def test_decision_tree_regressor_simple(simple_reg_data):
    X, y = simple_reg_data
    reg = DecisionTreeRegressor(max_depth=1)
    reg.fit(X, y)
    assert reg._is_fitted
    assert reg.root is not None
    assert not reg.root.is_leaf_node()
    y_pred = reg.predict(X)
    assert r2_score(y, y_pred) > 0.7
    reg_perfect = DecisionTreeRegressor(max_depth=5, min_samples_split=2)
    reg_perfect.fit(X, y)
    y_pred_perfect = reg_perfect.predict(X)
    assert r2_score(y, y_pred_perfect) > 0.99
