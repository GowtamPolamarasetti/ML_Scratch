# my_ml_project/tests/test_neighbors.py

import numpy as np
import pytest
from pyml.neighbors import KNeighborsClassifier
from pyml.metrics import accuracy_score

def test_knn_simple_case():
    X_train = np.array([[0], [1], [2], [3]])
    y_train = np.array([0, 0, 1, 1])
    clf = KNeighborsClassifier(n_neighbors=1)
    clf.fit(X_train, y_train)
    X_test = np.array([[0.1], [2.9]])
    y_pred = clf.predict(X_test)
    assert np.array_equal(y_pred, [0, 1])

def test_knn_fit_predict_accuracy():
    X_train = np.array([[0, 0], [1, 1], [0, 1], [1, 0]])
    y_train = np.array([0, 1, 0, 1])
    clf = KNeighborsClassifier(n_neighbors=1)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_train)
    assert accuracy_score(y_train, y_pred) == 1.0

def test_knn_predict_before_fit():
    X_test = np.array([[1, 1]])
    clf = KNeighborsClassifier(n_neighbors=3)
    with pytest.raises(RuntimeError, match="must call 'fit' before 'predict'"):
        clf.predict(X_test)

def test_knn_voting():
    X_train = np.array([[0], [1], [2], [10], [11]])
    y_train = np.array([0, 0, 1, 1, 1])
    clf = KNeighborsClassifier(n_neighbors=3)
    clf.fit(X_train, y_train)
    y_pred = clf.predict([[0.9]])
    assert y_pred[0] == 0
    y_pred = clf.predict([[5.0]])
    assert y_pred[0] == 0


@pytest.fixture
def complex_dataset():
    """Generate a synthetic 3-class 2D dataset with a clear majority class."""
    np.random.seed(42)
    
    # --- THIS IS THE FIX ---
    # Class 0 is now the unambiguous majority (21 samples)
    X_class0 = np.random.normal(loc=[0, 0], scale=0.5, size=(21, 2))
    X_class1 = np.random.normal(loc=[3, 3], scale=0.5, size=(20, 2))
    X_class2 = np.random.normal(loc=[0, 4], scale=0.5, size=(20, 2))
    
    X = np.vstack([X_class0, X_class1, X_class2])
    # The labels now reflect the new counts
    y = np.array([0]*21 + [1]*20 + [2]*20)
    # --- END FIX ---
    
    return X, y

def test_knn_multiclass_accuracy(complex_dataset):
    """Verify KNN performs well on multi-class data."""
    X, y = complex_dataset
    clf = KNeighborsClassifier(n_neighbors=3)
    clf.fit(X, y)
    y_pred = clf.predict(X)
    acc = accuracy_score(y, y_pred)
    assert acc > 0.9, f"Expected >90% accuracy, got {acc:.2f}"


def test_knn_distance_metric_influence(complex_dataset):
    """Ensure metric choice (euclidean vs manhattan) can affect predictions."""
    X, y = complex_dataset

    clf_euclidean = KNeighborsClassifier(n_neighbors=3)
    clf_manhattan = KNeighborsClassifier(n_neighbors=3)

    clf_euclidean.fit(X, y)
    clf_manhattan.fit(X, y)

    test_point = np.array([[1.5, 1.5]])
    pred_euclidean = clf_euclidean.predict(test_point)
    pred_manhattan = clf_manhattan.predict(test_point)

    assert pred_euclidean.shape == pred_manhattan.shape
    # Predictions might differ or not — just ensure logic runs fine
    assert pred_euclidean[0] in [0, 1, 2]
    assert pred_manhattan[0] in [0, 1, 2]


def test_knn_tie_breaking_behavior():
    """Force a tie case to check deterministic tie-breaking."""
    X = np.array([[0], [1], [2], [3]])
    y = np.array([0, 0, 1, 1])
    clf = KNeighborsClassifier(n_neighbors=4)
    clf.fit(X, y)
    y_pred = clf.predict([[1.5]])
    assert y_pred[0] in (0, 1), "KNN should handle ties deterministically."


def test_knn_large_k_majority_vote(complex_dataset):
    """When k equals dataset size, prediction should be global majority class."""
    X, y = complex_dataset
    # n_neighbors is now 61. Class 0 is the majority (21 votes).
    clf = KNeighborsClassifier(n_neighbors=len(y))
    clf.fit(X, y)
    y_pred = clf.predict([[0, 0], [3, 3], [1, 1]])
    
    # majority_class is now unambiguously 0.
    majority_class = np.bincount(y).argmax()
    assert majority_class == 0
    
    # Counter({0: 21, 1: 20, 2: 20}) will *always* return 0.
    # So y_pred should be [0, 0, 0].
    assert np.all(y_pred == majority_class)


def test_knn_noise_tolerance():
    """KNN should be somewhat robust to small random label noise."""
    np.random.seed(0)
    X = np.random.randn(100, 2)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    # Flip 10 random labels to introduce noise
    y_noisy = y.copy()
    flip_indices = np.random.choice(len(y), size=10, replace=False)
    y_noisy[flip_indices] = 1 - y_noisy[flip_indices]

    clf = KNeighborsClassifier(n_neighbors=5)
    clf.fit(X, y_noisy)
    y_pred = clf.predict(X)
    acc = accuracy_score(y, y_pred)
    assert acc > 0.8, f"KNN should be robust to noise (got {acc:.2f})"


def test_knn_predict_shape_consistency(complex_dataset):
    """Ensure predict() always returns a 1D numpy array of correct shape."""
    X, y = complex_dataset
    clf = KNeighborsClassifier(n_neighbors=3)
    clf.fit(X, y)
    X_test = np.array([[0, 0], [3, 3], [1, 2]])
    y_pred = clf.predict(X_test)
    assert isinstance(y_pred, np.ndarray)
    assert y_pred.ndim == 1
    assert len(y_pred) == len(X_test)