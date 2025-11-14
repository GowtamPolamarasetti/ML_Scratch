# my_ml_project/pyml/neighbors/_knn.py

import numpy as np
from collections import Counter
from ..base import BaseEstimator
from ..utils._validation import _check_X_y, _check_X
from ..utils._math import euclidean_distance

class KNeighborsClassifier(BaseEstimator):
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors

    def fit(self, X, y):
        X, y = _check_X_y(X, y)
        self.X_train_ = X
        self.y_train_ = y
        return self

    def predict(self, X):
        if not hasattr(self, 'X_train_'):
            raise RuntimeError("You must call 'fit' before 'predict'.")
        X = _check_X(X)
        predictions = [self._predict_one(x) for x in X]
        return np.asarray(predictions)

    def _predict_one(self, x):
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train_]
        k_indices = np.argsort(distances)[:self.n_neighbors]
        k_nearest_labels = [self.y_train_[i] for i in k_indices]
        most_common = Counter(k_nearest_labels).most_common(1)[0][0]
        return most_common
