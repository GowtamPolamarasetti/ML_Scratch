# my_ml_project/tests/test_decomposition.py

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pyml.decomposition import PCA

def test_pca():
    np.random.seed(42)
    X = np.dot(np.random.rand(100, 2), np.random.rand(2, 3))
    X = X + np.random.rand(100, 3) * 0.1
    pca = PCA(n_components=2)
    pca.fit(X)
    assert pca._is_fitted
    assert pca.mean_.shape == (3,)
    assert pca.components_.shape == (2, 3)
    assert pca.explained_variance_.shape == (2,)
    assert pca.explained_variance_ratio_.shape == (2,)
    assert pca.explained_variance_[0] > pca.explained_variance_[1]
    assert np.sum(pca.explained_variance_ratio_) > 0.8
    X_transformed = pca.transform(X)
    assert X_transformed.shape == (100, 2)
    X_transformed_ft = pca.fit_transform(X)
    assert_allclose(X_transformed, X_transformed_ft)
