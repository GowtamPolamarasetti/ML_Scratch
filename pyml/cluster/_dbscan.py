# my_ml_project/pyml/cluster/_dbscan.py

import numpy as np
from ..base import BaseEstimator
from ..utils._validation import _check_X
from ..utils._math import euclidean_distance

class DBSCAN(BaseEstimator):
    _NOISE = -1
    _UNVISITED = -2

    def __init__(self, eps=0.5, min_samples=5):
        self.eps = eps
        self.min_samples = min_samples
        self.labels_ = None
        self._is_fitted = False

    def _get_neighbors(self, X, sample_idx):
        neighbors = []
        p1 = X[sample_idx]
        for i in range(X.shape[0]):
            if i == sample_idx:
                continue
            p2 = X[i]
            if euclidean_distance(p1, p2) <= self.eps:
                neighbors.append(i)
        return neighbors

    def _expand_cluster(self, X, labels, sample_idx, cluster_id):
        neighbors = self._get_neighbors(X, sample_idx)
        if len(neighbors) < self.min_samples:
            labels[sample_idx] = self._NOISE
            return False
        labels[sample_idx] = cluster_id
        queue = neighbors[:]
        i = 0
        while i < len(queue):
            neighbor_idx = queue[i]
            i += 1
            if labels[neighbor_idx] == self._NOISE:
                labels[neighbor_idx] = cluster_id
            elif labels[neighbor_idx] == self._UNVISITED:
                labels[neighbor_idx] = cluster_id
                neighbor_neighbors = self._get_neighbors(X, neighbor_idx)
                if len(neighbor_neighbors) >= self.min_samples:
                    for nn_idx in neighbor_neighbors:
                        if labels[nn_idx] == self._UNVISITED:
                            if nn_idx not in queue:
                                queue.append(nn_idx)
        return True

    def fit(self, X, y=None):
        X = _check_X(X)
        n_samples = X.shape[0]
        self.labels_ = np.full(n_samples, self._UNVISITED, dtype=int)
        cluster_id = 0
        for i in range(n_samples):
            if self.labels_[i] != self._UNVISITED:
                continue
            if self._expand_cluster(X, self.labels_, i, cluster_id):
                cluster_id += 1
        self._is_fitted = True
        return self
