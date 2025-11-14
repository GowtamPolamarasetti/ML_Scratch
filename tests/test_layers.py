# my_ml_project/tests/test_neural_network/test_layers.py

import numpy as np
from numpy.testing import assert_allclose
from pyml.neural_network._layers import Dense, Flatten

def test_dense_shapes():
    n_samples, n_input, n_output = 32, 784, 128
    X = np.random.randn(n_samples, n_input)
    layer = Dense(n_input, n_output)
    y = layer.forward(X)
    assert y.shape == (n_samples, n_output)
    grad_y = np.random.randn(n_samples, n_output)
    grad_x = layer.backward(grad_y)
    assert grad_x.shape == (n_samples, n_input)
    assert layer.grads['W'].shape == (n_input, n_output)
    assert layer.grads['b'].shape == (n_output,)

def test_flatten_shapes():
    n_samples, C, H, W = 32, 3, 28, 28
    X = np.random.randn(n_samples, C, H, W)
    layer = Flatten()
    y = layer.forward(X)
    assert y.shape == (n_samples, C * H * W)
    grad_y = np.random.randn(n_samples, C * H * W)
    grad_x = layer.backward(grad_y)
    assert grad_x.shape == (n_samples, C, H, W)
