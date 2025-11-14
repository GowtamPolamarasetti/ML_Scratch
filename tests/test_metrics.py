# my_ml_project/tests/test_metrics.py

import numpy as np
import pytest
from numpy.testing import assert_array_equal, assert_allclose

from pyml.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    mean_squared_error,
    r2_score
)

def test_accuracy_score():
    y_true = [0, 1, 1, 0, 1, 1]
    y_pred = [0, 0, 1, 0, 1, 1]
    assert accuracy_score(y_true, y_pred) == 5/6

def test_accuracy_perfect():
    y_true = [0, 1, 2, 3]
    y_pred = [0, 1, 2, 3]
    assert accuracy_score(y_true, y_pred) == 1.0

def test_accuracy_none():
    y_true = [0, 1, 0, 1]
    y_pred = [1, 0, 1, 0]
    assert accuracy_score(y_true, y_pred) == 0.0

def test_precision_recall_f1():
    y_true = [0, 1, 1, 0, 1, 0]
    y_pred = [0, 1, 0, 0, 1, 1]
    assert_allclose(precision_score(y_true, y_pred), 2/3)
    assert_allclose(recall_score(y_true, y_pred), 2/3)
    assert_allclose(f1_score(y_true, y_pred), 2/3)

def test_f1_perfect():
    y_true = [0, 1, 1, 0]
    y_pred = [0, 1, 1, 0]
    assert f1_score(y_true, y_pred) == 1.0

def test_precision_recall_f1_zero_division():
    y_true = [0, 0, 0]
    y_pred = [0, 0, 0]
    assert precision_score(y_true, y_pred, zero_division=0.0) == 0.0
    assert recall_score(y_true, y_pred, zero_division=0.0) == 0.0
    assert f1_score(y_true, y_pred, zero_division=0.0) == 0.0
    assert precision_score(y_true, y_pred, zero_division=1.0) == 1.0
    y_true_no_pos = [0, 0]
    y_pred_has_pos = [0, 1]
    assert precision_score(y_true_no_pos, y_pred_has_pos) == 0.0
    assert recall_score(y_true_no_pos, y_pred_has_pos, zero_division=0.0) == 0.0

def test_mean_squared_error():
    y_true = [1.0, 2.0, 3.0]
    y_pred = [1.0, 2.0, 3.0]
    assert mean_squared_error(y_true, y_pred) == 0.0
    y_pred = [1.0, 2.0, 2.0]
    assert_allclose(mean_squared_error(y_true, y_pred), 1/3)
    y_pred = [2.0, 3.0, 4.0]
    assert_allclose(mean_squared_error(y_true, y_pred), 1.0)

def test_r2_score():
    y_true = [1, 2, 3]
    y_pred_perfect = [1, 2, 3]
    assert r2_score(y_true, y_pred_perfect) == 1.0
    y_pred_mean = [2, 2, 2]
    assert r2_score(y_true, y_pred_mean) == 0.0
    y_pred_worse = [3, 1, 2]
    assert r2_score(y_true, y_pred_worse) == -2.0
