import numpy as np
from pyml.model_selection import train_test_split

def test_train_test_split_shapes():
    X = np.arange(100).reshape(50, 2)
    y = np.arange(50)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    assert X_train.shape == (40, 2)
    assert y_train.shape == (40,)
    assert X_test.shape == (10, 2)
    assert y_test.shape == (10,)

def test_train_test_split_random_state():
    X = np.arange(20).reshape(10, 2)
    y = np.arange(10)
    
    # Call 1
    X_train1, X_test1, y_train1, y_test1 = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Call 2
    X_train2, X_test2, y_train2, y_test2 = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Results should be identical
    assert np.array_equal(X_train1, X_train2)
    assert np.array_equal(y_train1, y_train2)
    assert np.array_equal(X_test1, X_test2)
    assert np.array_equal(y_test1, y_test2)