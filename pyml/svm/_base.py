# my_ml_project/pyml/svm/_base.py

import numpy as np
from ..base import BaseEstimator
from ..utils._validation import _check_X_y, _check_X

class LinearSVC(BaseEstimator):
    def __init__(self, C=1.0, learning_rate=0.001, n_iterations=1000, random_state=None):
        self.C = C
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.random_state = random_state
        self.coef_ = None
        self.intercept_ = None
        self.history_ = {'cost': []}
        self._is_fitted = False
        if random_state:
            np.random.seed(random_state)

    def fit(self, X, y):
        X, y = _check_X_y(X, y)
        n_samples, n_features = X.shape
        classes = np.unique(y)
        if len(classes) != 2:
            raise ValueError("LinearSVC supports only binary classification.")
        self._class_map = {classes[0]: -1, classes[1]: 1}
        y_mapped = np.array([self._class_map[val] for val in y])
        self.coef_ = np.random.randn(n_features) * 0.01
        self.intercept_ = 0.0
        for i in range(self.n_iterations):
            linear_output = np.dot(X, self.coef_) + self.intercept_
            margin = y_mapped * linear_output
            hinge_loss = np.maximum(0, 1 - margin)
            cost = (0.5 * np.dot(self.coef_, self.coef_)) + (self.C * np.mean(hinge_loss))
            self.history_['cost'].append(cost)
            sv_indices = np.where(margin < 1)[0]
            if len(sv_indices) == 0:
                break
            X_sv = X[sv_indices]
            y_sv = y_mapped[sv_indices]
            dw = self.coef_ - (self.C / n_samples) * np.dot(X_sv.T, y_sv)
            db = - (self.C / n_samples) * np.sum(y_sv)
            self.coef_ -= self.learning_rate * dw
            self.intercept_ -= self.learning_rate * db
        self._is_fitted = True
        self._inv_class_map = {v: k for k, v in self._class_map.items()}
        return self

    def decision_function(self, X):
        if not self._is_fitted:
            raise RuntimeError("LinearSVC must be fitted before predicting.")
        X = _check_X(X)
        return np.dot(X, self.coef_) + self.intercept_

    def predict(self, X):
        scores = self.decision_function(X)
        predictions_mapped = np.sign(scores)
        predictions_mapped[predictions_mapped == 0] = -1
        return np.array([self._inv_class_map[val] for val in predictions_mapped])
