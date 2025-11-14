# my_ml_project/pyml/ensemble/_gradient_boosting.py

import numpy as np
from ..base import BaseEstimator
from ..utils._validation import _check_X_y, _check_X
from ..utils._math import sigmoid
from ..tree import DecisionTreeRegressor

class GradientBoostingClassifier(BaseEstimator):
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3, random_state=None):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.random_state = random_state
        self.trees_ = []
        self.initial_prediction_ = None
        self._classes = None
        self._is_fitted = False

    def fit(self, X, y):
        X, y = _check_X_y(X, y)
        n_samples = X.shape[0]
        self._classes = np.unique(y)
        if len(self._classes) != 2:
            raise ValueError("This GradientBoosting implementation only supports binary classification.")
        p = np.mean(y)
        self.initial_prediction_ = np.log(p / (1 - p))
        F_m = np.full(n_samples, self.initial_prediction_)
        self.trees_ = []
        for _ in range(self.n_estimators):
            p_m = sigmoid(F_m)
            residuals = y - p_m
            tree = DecisionTreeRegressor(max_depth=self.max_depth, min_samples_split=2)
            tree.fit(X, residuals)
            gamma_m = tree.predict(X)
            F_m += self.learning_rate * gamma_m
            self.trees_.append(tree)
        self._is_fitted = True
        return self

    def decision_function(self, X):
        if not self._is_fitted:
            raise RuntimeError("GradientBoostingClassifier must be fitted before predicting.")
        X = _check_X(X)
        F_m = np.full(X.shape[0], self.initial_prediction_)
        for tree in self.trees_:
            F_m += self.learning_rate * tree.predict(X)
        return F_m

    def predict_proba(self, X):
        log_odds = self.decision_function(X)
        proba_class_1 = sigmoid(log_odds)
        proba_class_0 = 1 - proba_class_1
        return np.vstack((proba_class_0, proba_class_1)).T

    def predict(self, X, threshold=0.5):
        proba_class_1 = self.predict_proba(X)[:, 1]
        return np.where(proba_class_1 >= threshold, self._classes[1], self._classes[0])
