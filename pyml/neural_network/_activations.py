# my_ml_project/pyml/neural_network/_activations.py

import numpy as np

class Activation:
    def __init__(self):
        self.input = None

    def forward(self, input_data):
        raise NotImplementedError

    def backward(self, output_gradient):
        raise NotImplementedError

class ReLU(Activation):
    def forward(self, input_data):
        self.input = input_data
        return np.maximum(0, input_data)

    def backward(self, output_gradient):
        input_gradient = output_gradient.copy()
        input_gradient[self.input <= 0] = 0
        return input_gradient

class Sigmoid(Activation):
    def forward(self, input_data):
        self.input = input_data
        self.output = 1 / (1 + np.exp(-input_data))
        return self.output

    def backward(self, output_gradient):
        sig_derivative = self.output * (1 - self.output)
        return output_gradient * sig_derivative

class Tanh(Activation):
    def forward(self, input_data):
        self.input = input_data
        self.output = np.tanh(input_data)
        return self.output

    def backward(self, output_gradient):
        tanh_derivative = 1 - self.output**2
        return output_gradient * tanh_derivative
