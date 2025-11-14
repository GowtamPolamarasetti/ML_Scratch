# my_ml_project/tests/test_neural_network/test_activations.py

import numpy as np
from numpy.testing import assert_allclose
from pyml.neural_network._activations import ReLU

def test_relu():
    relu = ReLU()
    X = np.array([[-2, -1, 0, 1, 2], [0.5, -0.5, 0, 1, -1]])

    y = relu.forward(X)
    expected_y = np.array([[0, 0, 0, 1, 2], [0.5, 0, 0, 1, 0]])
    assert_allclose(y, expected_y)

    grad_y = np.array([[1, 1, 1, 1, 1], [-1, -1, -1, -1, -1]])
    grad_x = relu.backward(grad_y)
    expected_grad_x = grad_y * (X > 0)
    assert_allclose(grad_x, expected_grad_x)
