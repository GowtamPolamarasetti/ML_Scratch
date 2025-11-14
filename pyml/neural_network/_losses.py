# my_ml_project/pyml/neural_network/_losses.py

import numpy as np

class Loss:
    def __init__(self):
        self.gradient = None

    def forward(self, y_pred, y_true):
        raise NotImplementedError

    def backward(self):
        raise NotImplementedError

class MeanSquaredError(Loss):
    def forward(self, y_pred, y_true):
        self.y_pred = y_pred
        self.y_true = y_true
        self.n_samples = y_pred.shape[0]
        loss = np.mean((y_pred - y_true)**2)
        return loss

    def backward(self):
        self.gradient = 2.0 * (self.y_pred - self.y_true) / self.n_samples
        return self.gradient

class SoftmaxCrossEntropyLoss(Loss):
    def forward(self, logits, y_true_indices):
        self.y_true_indices = y_true_indices
        self.n_samples = logits.shape[0]
        logits_stable = logits - np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(logits_stable)
        self.probas = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        correct_class_log_probas = -np.log(self.probas[range(self.n_samples), y_true_indices])
        loss = np.mean(correct_class_log_probas)
        return loss

    def backward(self):
        self.gradient = self.probas.copy()
        self.gradient[range(self.n_samples), self.y_true_indices] -= 1
        self.gradient = self.gradient / self.n_samples
        return self.gradient
