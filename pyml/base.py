# my_ml_project/pyml/base.py

class BaseEstimator:
    def __init__(self):
        pass

    def get_params(self, deep=True):
        params = {}
        for key, value in self.__dict__.items():
            if not key.startswith('_') and not key.endswith('_'):
                params[key] = value
        return params

    def set_params(self, **params):
        if not params:
            return self
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Invalid parameter {key} for estimator {self.__class__.__name__}")
        return self

    def fit(self, X, y=None):
        raise NotImplementedError("Each estimator must implement its own 'fit' method.")

    def predict(self, X):
        raise NotImplementedError("Each estimator must implement its own 'predict' method.")


class TransformerMixin:
    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y)
        return self.transform(X)

    def transform(self, X):
        raise NotImplementedError("Each transformer must implement its own 'transform' method.")
