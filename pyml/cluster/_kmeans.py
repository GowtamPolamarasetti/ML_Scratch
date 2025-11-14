# my_ml_project/pyml/cluster/_kmeans.py

import numpy as np
from ..base import BaseEstimator
from ..utils._validation import _check_X
from ..utils._math import euclidean_distance

class KMeans(BaseEstimator):
    def __init__(self, n_clusters=8, n_init=10, max_iter=300, random_state=None):
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.max_iter = max_iter
        self.random_state = random_state
        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = None
        self._is_fitted = False
        if random_state:
            np.random.seed(random_state)

    def _init_centroids(self, X):
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, self.n_clusters, replace=False)
        return X[indices]

    def _assign_clusters(self, X, centroids):
        n_samples = X.shape[0]
        distances = np.zeros((n_samples, self.n_clusters))
        for k in range(self.n_clusters):
            distances[:, k] = np.array([euclidean_distance(x, centroids[k]) for x in X])
        return np.argmin(distances, axis=1)

    def _update_centroids(self, X, labels):
        n_features = X.shape[1]
        centroids = np.zeros((self.n_clusters, n_features))
        for k in range(self.n_clusters):
            cluster_samples = X[labels == k]
            if len(cluster_samples) > 0:
                centroids[k] = np.mean(cluster_samples, axis=0)
            else:
                centroids[k] = self._init_centroids(X)[0]
        return centroids

    def _calculate_inertia(self, X, labels, centroids):
        inertia = 0
        for k in range(self.n_clusters):
            cluster_samples = X[labels == k]
            if len(cluster_samples) > 0:
                inertia += np.sum((cluster_samples - centroids[k])**2)
        return inertia

    def fit(self, X, y=None):
        X = _check_X(X)
        best_inertia = np.inf
        best_centroids = None
        best_labels = None
        for _ in range(self.n_init):
            centroids = self._init_centroids(X)
            for i in range(self.max_iter):
                labels = self._assign_clusters(X, centroids)
                new_centroids = self._update_centroids(X, labels)
                if np.allclose(centroids, new_centroids):
                    break
                centroids = new_centroids
            inertia = self._calculate_inertia(X, labels, centroids)
            if inertia < best_inertia:
                best_inertia = inertia
                best_centroids = centroids
                best_labels = labels
        self.cluster_centers_ = best_centroids
        self.labels_ = best_labels
        self.inertia_ = best_inertia
        self._is_fitted = True
        return self

    def predict(self, X):
        if not self._is_fitted:
            raise RuntimeError("KMeans must be fitted before predicting.")
        X = _check_X(X)
        return self._assign_clusters(X, self.cluster_centers_)
