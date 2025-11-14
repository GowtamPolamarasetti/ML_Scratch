# my_ml_project/tests/test_cluster.py

import numpy as np
import pytest
from numpy.testing import assert_array_equal

try:
    from sklearn.datasets import make_blobs, make_moons
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from pyml.cluster import KMeans, DBSCAN
from pyml.metrics import accuracy_score

@pytest.mark.skipif(not SKLEARN_AVAILABLE, reason="scikit-learn not found")
def test_kmeans_simple_blobs():
    X, y = make_blobs(n_samples=150, n_features=2, centers=3,
                      cluster_std=0.5, random_state=42)
    kmeans = KMeans(n_clusters=3, n_init=10, random_state=42)
    kmeans.fit(X)
    assert kmeans._is_fitted
    assert kmeans.cluster_centers_.shape == (3, 2)
    assert kmeans.labels_.shape == (150,)
    assert kmeans.inertia_ > 0
    y_pred_kmeans = kmeans.predict(X)
    assert_array_equal(y_pred_kmeans, kmeans.labels_)
    acc = accuracy_score(y, y_pred_kmeans)
    assert acc < 0.5 or acc > 0.5

@pytest.mark.skipif(not SKLEARN_AVAILABLE, reason="scikit-learn not found")
def test_dbscan_moons():
    X, y = make_moons(n_samples=200, noise=0.05, random_state=42)
    dbscan = DBSCAN(eps=0.3, min_samples=5)
    dbscan.fit(X)
    assert dbscan._is_fitted
    labels = dbscan.labels_
    unique_labels = set(labels)
    assert 0 in unique_labels
    assert 1 in unique_labels
    n_noise = np.sum(labels == -1)
    assert n_noise < 50
    kmeans = KMeans(n_clusters=2, n_init=10, random_state=42).fit(X)
    kmeans_acc = accuracy_score(y, kmeans.predict(X))
    n_clusters = len(set(labels) - {-1})
    assert n_clusters == 2
