# my_ml_project/pyml/ensemble/_boosting.py

import numpy as np
from ..base import BaseEstimator
from ..utils._validation import _check_X_y, _check_X
from ..tree import DecisionTreeClassifier

class AdaBoostClassifier(BaseEstimator):
    def __init__(self, n_estimators=50, learning_rate=1.0, random_state=None):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.estimators_ = []
        self._classes = None
        self._is_fitted = False
        if random_state:
            np.random.seed(random_state)

    def fit(self, X, y):
        X, y = _check_X_y(X, y)
        n_samples = X.shape[0]
        self._classes = np.unique(y)
        n_classes = len(self._classes)
        if n_classes < 2:
            raise ValueError("AdaBoost requires at least 2 classes.")
        if n_classes != 2:
            raise ValueError("This AdaBoost implementation only supports binary classification.")
        class_map = {self._classes[0]: -1, self._classes[1]: 1}
        y_numeric = np.array([class_map[val] for val in y])
        w = np.full(n_samples, 1 / n_samples)
        self.estimators_ = []
        for i in range(self.n_estimators):
            indices = np.random.choice(n_samples, size=n_samples, replace=True, p=w)
            X_sample, y_sample_numeric = X[indices], y_numeric[indices]
            tree = DecisionTreeClassifier(max_depth=1)
            tree.fit(X_sample, y_sample_numeric)
            y_pred_numeric = tree.predict(X)
            misclassified = (y_pred_numeric != y_numeric)
            err = np.sum(w[misclassified])
            if err == 0:
                alpha = self.learning_rate * (np.log(1.0 / 1e-10))
                self.estimators_.append((tree, alpha))
                break
            if err >= 0.5:
                break
            alpha = self.learning_rate * 0.5 * np.log((1 - err) / err)
            w = w * np.exp(-alpha * y_numeric * y_pred_numeric)
            w = w / np.sum(w)
            self.estimators_.append((tree, alpha))
        self._is_fitted = True
        self._inv_class_map = {v: k for k, v in class_map.items()}
        return self

    def decision_function(self, X):
        if not self._is_fitted:
            raise RuntimeError("AdaBoostClassifier must be fitted before predicting.")
        X = _check_X(X)
        final_preds = np.zeros(X.shape[0])
        for tree, alpha in self.estimators_:
            final_preds += alpha * tree.predict(X)
        return final_preds

    def predict(self, X):
        scores = self.decision_function(X)
        predictions_mapped = np.sign(scores).astype(int)
        predictions_mapped[predictions_mapped == 0] = -1
        return np.array([self._inv_class_map[val] for val in predictions_mapped])
