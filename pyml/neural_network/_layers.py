# my_ml_project/pyml/neural_network/_layers.py

import numpy as np
from ._activations import Activation

class Layer:
    def __init__(self):
        self.input = None
        self.params = {}
        self.grads = {}

    def forward(self, input_data):
        raise NotImplementedError

    def backward(self, output_gradient):
        raise NotImplementedError

class Dense(Layer):
    def __init__(self, n_input, n_output, random_state=None):
        super().__init__()
        if random_state:
            np.random.seed(random_state)
        self.params['W'] = np.random.randn(n_input, n_output) * np.sqrt(1 / n_input)
        self.params['b'] = np.zeros(n_output)

    def forward(self, input_data):
        self.input = input_data
        return np.dot(self.input, self.params['W']) + self.params['b']

    def backward(self, output_gradient):
        self.grads['W'] = np.dot(self.input.T, output_gradient)
        self.grads['b'] = np.sum(output_gradient, axis=0)
        input_gradient = np.dot(output_gradient, self.params['W'].T)
        return input_gradient

class ActivationLayer(Layer):
    def __init__(self, activation: Activation):
        super().__init__()
        self.activation = activation

    def forward(self, input_data):
        self.input = input_data
        return self.activation.forward(input_data)

    def backward(self, output_gradient):
        return self.activation.backward(output_gradient)

class Flatten(Layer):
    def __init__(self):
        super().__init__()
        self.input_shape = None

    def forward(self, input_data):
        self.input_shape = input_data.shape
        n_samples = self.input_shape[0]
        return input_data.reshape(n_samples, -1)

    def backward(self, output_gradient):
        return output_gradient.reshape(self.input_shape)
