# my_ml_project/pyml/ensemble/_forest.py

import numpy as np
from collections import Counter
from ..base import BaseEstimator
from ..utils._validation import _check_X_y, _check_X
from ..tree import DecisionTreeClassifier

class RandomForestClassifier(BaseEstimator):
    def __init__(self, n_estimators=100, max_depth=10, min_samples_split=2, 
                 max_features=None, random_state=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.random_state = random_state
        self.trees_ = []
        self._is_fitted = False
        if random_state:
            np.random.seed(random_state)

    def _bootstrap_sample(self, X, y):
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        return X[indices], y[indices]

    def fit(self, X, y):
        X, y = _check_X_y(X, y)
        n_samples, n_features = X.shape
        if self.max_features is None:
            self.max_features = n_features
        elif not (0 < self.max_features <= n_features):
            raise ValueError("max_features must be in (0, n_features]")
        self.trees_ = []
        for i in range(self.n_estimators):
            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split
            )
            X_sample, y_sample = self._bootstrap_sample(X, y)
            tree.fit(X_sample, y_sample)
            self.trees_.append(tree)
        self._is_fitted = True
        return self

    def _majority_vote(self, y_preds_all_trees):
        y_preds = y_preds_all_trees.T
        predictions = []
        for sample_preds in y_preds:
            vote_counts = Counter(sample_preds)
            majority_vote = vote_counts.most_common(1)[0][0]
            predictions.append(majority_vote)
        return np.array(predictions)

    def predict(self, X):
        if not self._is_fitted:
            raise RuntimeError("RandomForestClassifier must be fitted before predicting.")
        X = _check_X(X)
        all_preds = np.array([tree.predict(X) for tree in self.trees_])
        return self._majority_vote(all_preds)
