# my_ml_project/pyml/tree/_tree.py

import numpy as np
from collections import Counter
from ..base import BaseEstimator
from ..utils._validation import _check_X_y, _check_X

class _Node:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, *, value=None, impurity=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.impurity = impurity
        self.value = value

    def is_leaf_node(self):
        return self.value is not None

class DecisionTreeClassifier(BaseEstimator):
    def __init__(self, max_depth=100, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None
        self._is_fitted = False

    def _gini_impurity(self, y):
        if y.shape[0] == 0:
            return 0
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / y.shape[0]
        return 1 - np.sum(probabilities**2)

    def _calculate_weighted_impurity(self, y_left, y_right):
        n = y_left.shape[0] + y_right.shape[0]
        if n == 0:
            return 0
        n_left, n_right = y_left.shape[0], y_right.shape[0]
        gini_left = self._gini_impurity(y_left)
        gini_right = self._gini_impurity(y_right)
        impurity = (n_left / n) * gini_left + (n_right / n) * gini_right
        return impurity

    def _find_best_split(self, X, y):
        n_samples, n_features = X.shape
        if n_samples <= 1:
            return None, None
        current_impurity = self._gini_impurity(y)
        best_gain = -1
        best_feature_index = None
        best_threshold = None
        for feature_index in range(n_features):
            thresholds = np.unique(X[:, feature_index])
            for threshold in thresholds:
                indices_left = np.where(X[:, feature_index] <= threshold)[0]
                indices_right = np.where(X[:, feature_index] > threshold)[0]
                if len(indices_left) == 0 or len(indices_right) == 0:
                    continue
                y_left, y_right = y[indices_left], y[indices_right]
                split_impurity = self._calculate_weighted_impurity(y_left, y_right)
                gain = current_impurity - split_impurity
                if gain > best_gain:
                    best_gain = gain
                    best_feature_index = feature_index
                    best_threshold = threshold
        return best_feature_index, best_threshold

    def _build_tree(self, X, y, depth=0):
        n_samples, _ = X.shape
        n_labels = len(np.unique(y))
        if (depth >= self.max_depth or 
            n_samples < self.min_samples_split or
            n_labels == 1):
            leaf_value = self._most_common_label(y)
            return _Node(value=leaf_value)
        feature_index, threshold = self._find_best_split(X, y)
        if feature_index is None:
            leaf_value = self._most_common_label(y)
            return _Node(value=leaf_value)
        indices_left = np.where(X[:, feature_index] <= threshold)[0]
        indices_right = np.where(X[:, feature_index] > threshold)[0]
        X_left, y_left = X[indices_left], y[indices_left]
        X_right, y_right = X[indices_right], y[indices_right]
        left_child = self._build_tree(X_left, y_left, depth + 1)
        right_child = self._build_tree(X_right, y_right, depth + 1)
        return _Node(feature_index=feature_index, threshold=threshold, left=left_child, right=right_child, impurity=self._gini_impurity(y))

    def _most_common_label(self, y):
        if len(y) == 0:
            return None
        counter = Counter(y)
        return counter.most_common(1)[0][0]

    def fit(self, X, y):
        X, y = _check_X_y(X, y)
        self.root = self._build_tree(X, y)
        self._is_fitted = True
        return self

    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value
        if x[node.feature_index] <= node.threshold:
            return self._traverse_tree(x, node.left)
        else:
            return self._traverse_tree(x, node.right)

    def predict(self, X):
        if not self._is_fitted:
            raise RuntimeError("DecisionTreeClassifier must be fitted before predicting.")
        X = _check_X(X)
        return np.array([self._traverse_tree(x, self.root) for x in X])

class DecisionTreeRegressor(BaseEstimator):
    def __init__(self, max_depth=100, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None
        self._is_fitted = False

    def _mse(self, y):
        if y.shape[0] == 0:
            return 0
        return np.mean((y - np.mean(y))**2)

    def _calculate_weighted_mse(self, y_left, y_right):
        n = y_left.shape[0] + y_right.shape[0]
        if n == 0:
            return 0
        n_left, n_right = y_left.shape[0], y_right.shape[0]
        mse_left = self._mse(y_left)
        mse_right = self._mse(y_right)
        impurity = (n_left / n) * mse_left + (n_right / n) * mse_right
        return impurity

    def _find_best_split(self, X, y):
        n_samples, n_features = X.shape
        if n_samples <= 1:
            return None, None
        current_mse = self._mse(y)
        best_gain = -np.inf
        best_feature_index = None
        best_threshold = None
        for feature_index in range(n_features):
            thresholds = np.unique(X[:, feature_index])
            for threshold in thresholds:
                indices_left = np.where(X[:, feature_index] <= threshold)[0]
                indices_right = np.where(X[:, feature_index] > threshold)[0]
                if len(indices_left) == 0 or len(indices_right) == 0:
                    continue
                y_left, y_right = y[indices_left], y[indices_right]
                split_mse = self._calculate_weighted_mse(y_left, y_right)
                gain = current_mse - split_mse
                if gain > best_gain:
                    best_gain = gain
                    best_feature_index = feature_index
                    best_threshold = threshold
        return best_feature_index, best_threshold

    def _build_tree(self, X, y, depth=0):
        n_samples, _ = X.shape
        if (depth >= self.max_depth or 
            n_samples < self.min_samples_split or
            self._mse(y) == 0):
            leaf_value = np.mean(y)
            return _Node(value=leaf_value)
        feature_index, threshold = self._find_best_split(X, y)
        if feature_index is None:
            leaf_value = np.mean(y)
            return _Node(value=leaf_value)
        indices_left = np.where(X[:, feature_index] <= threshold)[0]
        indices_right = np.where(X[:, feature_index] > threshold)[0]
        X_left, y_left = X[indices_left], y[indices_left]
        X_right, y_right = X[indices_right], y[indices_right]
        left_child = self._build_tree(X_left, y_left, depth + 1)
        right_child = self._build_tree(X_right, y_right, depth + 1)
        return _Node(feature_index=feature_index, threshold=threshold, left=left_child, right=right_child, impurity=self._mse(y))

    def fit(self, X, y):
        X, y = _check_X_y(X, y)
        self.root = self._build_tree(X, y)
        self._is_fitted = True
        return self

    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value
        if x[node.feature_index] <= node.threshold:
            return self._traverse_tree(x, node.left)
        else:
            return self._traverse_tree(x, node.right)

    def predict(self, X):
        if not self._is_fitted:
            raise RuntimeError("DecisionTreeRegressor must be fitted before predicting.")
        X = _check_X(X)
        return np.array([self._traverse_tree(x, self.root) for x in X])
