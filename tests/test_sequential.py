import numpy as np
import pytest
from numpy.testing import assert_allclose

from pyml.neural_network import Sequential, Dense, ActivationLayer, Tanh 
from pyml.neural_network._losses import SoftmaxCrossEntropyLoss, MeanSquaredError
from pyml.neural_network._optimizers import Adam, SGD
from pyml.metrics import accuracy_score

def test_sequential_xor_solver():
    
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 1, 1, 0])
    
    model = Sequential()
    model.add(Dense(n_input=2, n_output=8, random_state=42))
    model.add(ActivationLayer(Tanh())) 
    model.add(Dense(n_input=8, n_output=2, random_state=42))

    model.compile(
        loss=SoftmaxCrossEntropyLoss(),
        optimizer_class=Adam,
        learning_rate=0.01
    )

    history = model.fit(X, y, epochs=200, batch_size=4)

    y_pred = model.predict(X)
    
    assert accuracy_score(y, y_pred) == 1.0
    
def test_sequential_regression_solver():
    
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
    
    assert history['loss'][0] > history['loss'][-1] * 10

    y_pred = model._forward(np.array([[6.0]]))
    assert_allclose(y_pred[0][0], 12.0, atol=1.0) 
