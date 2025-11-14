# my_ml_project/tests/test_utils.py

import numpy as np
import pytest
from numpy.testing import assert_array_equal, assert_allclose

from pyml.utils._validation import _check_X_y, _check_X
from pyml.utils._math import euclidean_distance, sigmoid, softmax

def test_check_X_y_basic():
    X = [[1, 2], [3, 4]]
    y = [1, 2]
    X_out, y_out = _check_X_y(X, y)
    assert isinstance(X_out, np.ndarray)
    assert isinstance(y_out, np.ndarray)
    assert_array_equal(X_out, np.array([[1, 2], [3, 4]]))
    assert_array_equal(y_out, np.array([1, 2]))

def test_check_X_for_predict():
    X = [[1, 2], [3, 4]]
    X_out = _check_X(X)
    assert isinstance(X_out, np.ndarray)
    assert X_out.shape == (2, 2)
    X_1d = [1, 2, 3]
    X_out_1d = _check_X(X_1d)
    assert X_out_1d.shape == (3, 1)

def test_check_X_y_mismatch():
    X = [[1, 2], [3, 4]]
    y = [1, 2, 3]
    match_str = "Inconsistent number of samples between X and y. Got 2 and 3."
    with pytest.raises(ValueError, match=match_str):
        _check_X_y(X, y)

def test_check_X_y_1d_X():
    X = [1, 2, 3]
    y = [1, 2, 3]
    X_out, y_out = _check_X_y(X, y)
    assert X_out.shape == (3, 1)

def test_check_X_y_empty():
    with pytest.raises(ValueError, match="at least 1 sample"):
        _check_X_y([], [])

def test_check_X_empty():
    with pytest.raises(ValueError, match="at least 1 sample"):
        _check_X([])

def test_euclidean_distance():
    x1 = np.array([0, 0, 0])
    x2 = np.array([3, 0, 4])
    assert euclidean_distance(x1, x2) == 5.0

def test_sigmoid():
    assert sigmoid(0) == 0.5
    assert_allclose(sigmoid(np.array([-1000, 1000])), np.array([0.0, 1.0]), atol=1e-9)
    assert_allclose(sigmoid(np.array([-2, 0, 2])), 
                    np.array([0.11920292, 0.5, 0.88079708]))

def test_softmax():
    z_1d = np.array([0, 1, 2])
    softmax_1d = softmax(z_1d.reshape(1, -1))
    assert softmax_1d.shape == (1, 3)
    assert_allclose(softmax_1d.sum(), 1.0)
    z_2d = np.array([[1, 2, 3], [1, 1, 1]])
    softmax_2d = softmax(z_2d)
    assert softmax_2d.shape == (2, 3)
    assert_allclose(np.sum(softmax_2d, axis=1), np.array([1.0, 1.0]))
    row1_exp = np.exp(np.array([1, 2, 3]) - 3)
    row1_sum = row1_exp.sum()
    assert_allclose(softmax_2d[0], row1_exp / row1_sum)
    assert_allclose(softmax_2d[1], np.array([1/3, 1/3, 1/3]))
