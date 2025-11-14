# my_ml_project/pyml/linear_model/_base.py

import numpy as np
from ..base import BaseEstimator
from ..utils._validation import _check_X_y, _check_X
from ..utils._math import sigmoid, softmax
from ..metrics._regression import mean_squared_error

class LinearRegression(BaseEstimator):
    def __init__(self, learning_rate=0.01, n_iterations=1000, random_state=None):
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
        self.coef_ = np.random.randn(n_features) * 0.01
        self.intercept_ = 0.0
        for i in range(self.n_iterations):
            y_pred = np.dot(X, self.coef_) + self.intercept_
            cost = mean_squared_error(y, y_pred) / 2
            self.history_['cost'].append(cost)
            errors = y_pred - y
            dw = (1 / n_samples) * np.dot(X.T, errors)
            db = (1 / n_samples) * np.sum(errors)
            self.coef_ -= self.learning_rate * dw
            self.intercept_ -= self.learning_rate * db
        self._is_fitted = True
        return self

    def predict(self, X):
        if not self._is_fitted:
            raise RuntimeError("LinearRegression must be fitted before predicting.")
        X = _check_X(X)
        y_pred = np.dot(X, self.coef_) + self.intercept_
        return y_pred

class LogisticRegression(BaseEstimator):
    def __init__(self, learning_rate=0.01, n_iterations=1000, random_state=None):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.random_state = random_state
        self.coef_ = None
        self.intercept_ = None
        self.history_ = {'cost': []}
        self._is_fitted = False
        if random_state:
            np.random.seed(random_state)

    def _compute_cost(self, y, y_pred):
        epsilon = 1e-9
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        cost = -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
        return cost

    def fit(self, X, y):
        X, y = _check_X_y(X, y)
        n_samples, n_features = X.shape
        if len(np.unique(y)) != 2:
            raise ValueError("LogisticRegression currently only supports binary classification (y must have 2 unique classes).")
        self.coef_ = np.zeros(n_features)
        self.intercept_ = 0.0
        for i in range(self.n_iterations):
            z = np.dot(X, self.coef_) + self.intercept_
            y_pred = sigmoid(z)
            cost = self._compute_cost(y, y_pred)
            self.history_['cost'].append(cost)
            errors = y_pred - y
            dw = (1 / n_samples) * np.dot(X.T, errors)
            db = (1 / n_samples) * np.sum(errors)
            self.coef_ -= self.learning_rate * dw
            self.intercept_ -= self.learning_rate * db
        self._is_fitted = True
        return self

    def predict_proba(self, X):
        if not self._is_fitted:
            raise RuntimeError("LogisticRegression must be fitted before predicting.")
        X = _check_X(X)
        z = np.dot(X, self.coef_) + self.intercept_
        proba_class_1 = sigmoid(z)
        proba_class_0 = 1 - proba_class_1
        return np.vstack((proba_class_0, proba_class_1)).T

    def predict(self, X, threshold=0.5):
        probas = self.predict_proba(X)[:, 1]
        return (probas >= threshold).astype(int)
