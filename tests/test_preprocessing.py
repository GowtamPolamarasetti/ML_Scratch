# my_ml_project/tests/test_preprocessing.py

import numpy as np
import pytest
from numpy.testing import assert_array_equal, assert_allclose

from pyml.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder

@pytest.fixture
def scaler_data():
    return np.array([[1.0, -1.0, 2.0],
                     [2.0,  0.0, 0.0],
                     [0.0,  1.0, 1.0]])

def test_standard_scaler(scaler_data):
    scaler = StandardScaler()
    scaler.fit(scaler_data)
    assert_allclose(scaler.mean_, np.array([1.0, 0.0, 1.0]))
    transformed = scaler.transform(scaler_data)
    assert_allclose(np.mean(transformed, axis=0), np.array([0.0, 0.0, 0.0]), atol=1e-9)
    assert_allclose(np.std(transformed, axis=0), np.array([1.0, 1.0, 1.0]), atol=1e-9)
    scaler2 = StandardScaler()
    transformed2 = scaler2.fit_transform(scaler_data)
    assert_allclose(transformed, transformed2)

def test_min_max_scaler(scaler_data):
    scaler = MinMaxScaler()
    scaler.fit(scaler_data)
    assert_allclose(scaler.data_min_, np.array([0.0, -1.0, 0.0]))
    assert_allclose(scaler.data_max_, np.array([2.0, 1.0, 2.0]))
    transformed = scaler.transform(scaler_data)
    expected = np.array([[0.5, 0.0, 1.0],
                         [1.0, 0.5, 0.0],
                         [0.0, 1.0, 0.5]])
    assert_allclose(transformed, expected)
    assert_allclose(np.min(transformed), 0.0)
    assert_allclose(np.max(transformed), 1.0)

def test_min_max_scaler_range():
    scaler = MinMaxScaler(feature_range=(-1, 1))
    X = np.array([[1.0], [2.0], [3.0]])
    transformed = scaler.fit_transform(X)
    expected = np.array([[-1.0], [0.0], [1.0]])
    assert_allclose(transformed, expected)

def test_label_encoder():
    le = LabelEncoder()
    y = ['cat', 'dog', 'cat', 'fish']
    le.fit(y)
    assert_array_equal(le.classes_, ['cat', 'dog', 'fish'])
    transformed = le.transform(y)
    assert_array_equal(transformed, [0, 1, 0, 2])
    inversed = le.inverse_transform(transformed)
    assert_array_equal(inversed, y)

def test_label_encoder_unseen():
    le = LabelEncoder()
    le.fit(['cat', 'dog'])
    with pytest.raises(ValueError, match="new labels"):
        le.transform(['cat', 'fish'])

def test_one_hot_encoder():
    ohe = OneHotEncoder()
    X = np.array([[0, 0], [1, 1], [0, 2]])
    ohe.fit(X)
    assert len(ohe.categories_) == 2
    assert_array_equal(ohe.categories_[0], [0, 1])
    assert_array_equal(ohe.categories_[1], [0, 1, 2])
    transformed = ohe.transform(X)
    expected = np.array([[1, 0, 1, 0, 0],
                         [0, 1, 0, 1, 0],
                         [1, 0, 0, 0, 1]])
    assert_array_equal(transformed, expected)
