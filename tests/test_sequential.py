# my_ml_project/tests/test_neural_network/test_sequential.py
# Test the full model on a simple problem (XOR)

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pyml.neural_network import Sequential, Dense, ActivationLayer, Tanh # <-- FIX: Was 'Activation'
from pyml.neural_network._losses import SoftmaxCrossEntropyLoss, MeanSquaredError
from pyml.neural_network._optimizers import Adam, SGD
from pyml.metrics import accuracy_score

def test_sequential_xor_solver():
    """Tests if the network can solve the XOR problem."""
    
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    # y = np.array([0, 1, 1, 0])
    
    # For classification, we need 2 output neurons
    y = np.array([0, 1, 1, 0])
    
    model = Sequential()
    model.add(Dense(n_input=2, n_output=8, random_state=42))
    model.add(ActivationLayer(Tanh())) # <-- FIX: Was 'Activation(Tanh())'
    model.add(Dense(n_input=8, n_output=2, random_state=42))
    
    # Compile with SoftmaxCrossEntropy (expects logits)
    model.compile(
        loss=SoftmaxCrossEntropyLoss(),
        optimizer_class=Adam,
        learning_rate=0.01
    )
    
    # Fit the model
    history = model.fit(X, y, epochs=200, batch_size=4)
    
    # Test predictions
    y_pred = model.predict(X)
    
    assert accuracy_score(y, y_pred) == 1.0
    
def test_sequential_regression_solver():
    """Tests if the network can solve a simple linear regression."""
    
    X = np.array([[1], [2], [3], [4], [5]])
    y = np.array([[2], [4], [6], [8], [10]])
    
    model = Sequential()
    model.add(Dense(n_input=1, n_output=1, random_state=42))
    
    model.compile(
        loss=MeanSquaredError(),
        optimizer_class=SGD,
        learning_rate=0.01
    )
    
    history = model.fit(X, y, epochs=100, batch_size=1)
    
    # Check if the loss decreased significantly
    assert history['loss'][0] > history['loss'][-1] * 10
    
    # Check if the final prediction is close
    y_pred = model._forward(np.array([[6.0]])) # Use _forward for raw value
    assert_allclose(y_pred[0][0], 12.0, atol=1.0) # Should be close to 12