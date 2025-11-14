# my_ml_project/tests/test_neural_network/test_losses.py

import numpy as np
from numpy.testing import assert_allclose
from pyml.neural_network._losses import SoftmaxCrossEntropyLoss

def test_softmax_cross_entropy():
    loss_fn = SoftmaxCrossEntropyLoss()
    logits = np.array([[2.0, 1.0, 0.1],
                       [1.0, 2.0, 0.1]])
    y_true = np.array([0, 1])
    loss = loss_fn.forward(logits, y_true)
    assert_allclose(loss, 0.4170, atol=1e-4)
    grad = loss_fn.backward()
    P = loss_fn.probas
    Y = np.array([[1, 0, 0], [0, 1, 0]])
    expected_grad = (P - Y) / 2.0
    assert_allclose(grad, expected_grad)
