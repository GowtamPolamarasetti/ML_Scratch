# my_ml_project/pyml/neural_network/_optimizers.py

import numpy as np

class Optimizer:
    def __init__(self, layers, learning_rate):
        self.layers = layers
        self.learning_rate = learning_rate

    def step(self):
        for layer in self.layers:
            if layer.params:
                self._update_layer(layer)

    def _update_layer(self, layer):
        raise NotImplementedError

class SGD(Optimizer):
    def __init__(self, layers, learning_rate=0.01):
        super().__init__(layers, learning_rate)

    def _update_layer(self, layer):
        for param_name in layer.params:
            layer.params[param_name] -= self.learning_rate * layer.grads[param_name]

class RMSprop(Optimizer):
    def __init__(self, layers, learning_rate=0.001, rho=0.9, epsilon=1e-8):
        super().__init__(layers, learning_rate)
        self.rho = rho
        self.epsilon = epsilon
        self.s = []
        for layer in self.layers:
            if layer.params:
                layer_s = {}
                for param_name, param_val in layer.params.items():
                    layer_s[param_name] = np.zeros_like(param_val)
                self.s.append(layer_s)
            else:
                self.s.append(None)

    def _update_layer(self, layer):
        layer_idx = self.layers.index(layer)
        for param_name in layer.params:
            grad = layer.grads[param_name]
            self.s[layer_idx][param_name] = self.rho * self.s[layer_idx][param_name] + (1 - self.rho) * (grad**2)
            update = self.learning_rate * grad / (np.sqrt(self.s[layer_idx][param_name]) + self.epsilon)
            layer.params[param_name] -= update

class Adam(Optimizer):
    def __init__(self, layers, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__(layers, learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0
        self.m = []
        self.v = []
        for layer in self.layers:
            if layer.params:
                layer_m = {}
                layer_v = {}
                for param_name, param_val in layer.params.items():
                    layer_m[param_name] = np.zeros_like(param_val)
                    layer_v[param_name] = np.zeros_like(param_val)
                self.m.append(layer_m)
                self.v.append(layer_v)
            else:
                self.m.append(None)
                self.v.append(None)

    def _update_layer(self, layer):
        self.t += 1
        layer_idx = self.layers.index(layer)
        for param_name in layer.params:
            grad = layer.grads[param_name]
            self.m[layer_idx][param_name] = self.beta1 * self.m[layer_idx][param_name] + (1 - self.beta1) * grad
            self.v[layer_idx][param_name] = self.beta2 * self.v[layer_idx][param_name] + (1 - self.beta2) * (grad**2)
            m_hat = self.m[layer_idx][param_name] / (1 - self.beta1**self.t)
            v_hat = self.v[layer_idx][param_name] / (1 - self.beta2**self.t)
            update = self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
            layer.params[param_name] -= update
