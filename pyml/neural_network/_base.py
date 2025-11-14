# my_ml_project/pyml/neural_network/_base.py

import numpy as np
from ._losses import Loss
from ._optimizers import Optimizer

class Sequential:
    def __init__(self):
        self.layers = []
        self.loss: Loss = None
        self.optimizer: Optimizer = None

    def add(self, layer):
        self.layers.append(layer)

    def compile(self, loss: Loss, optimizer_class, **optimizer_kwargs):
        self.loss = loss
        self.optimizer = optimizer_class(self.layers, **optimizer_kwargs)

    def _forward(self, X):
        output = X
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def _backward(self, loss_gradient):
        grad = loss_gradient
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
    
    def _get_batches(self, X, y, batch_size, shuffle=True):
        n_samples = X.shape[0]
        indices = np.arange(n_samples)
        if shuffle:
            np.random.shuffle(indices)
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            batch_indices = indices[start:end]
            yield X[batch_indices], y[batch_indices]
            
    def _to_one_hot(self, y, n_classes):
        one_hot = np.zeros((y.shape[0], n_classes))
        one_hot[np.arange(y.shape[0]), y] = 1
        return one_hot

    def fit(self, X_train, y_train, epochs, batch_size, validation_data=None):
        if self.loss is None or self.optimizer is None:
            raise RuntimeError("Must call .compile() before .fit()")
        n_samples = X_train.shape[0]
        n_classes = len(np.unique(y_train))
        history = {'loss': [], 'val_loss': [], 'val_acc': []}
        for epoch in range(epochs):
            epoch_loss = 0
            for X_batch, y_batch in self._get_batches(X_train, y_train, batch_size):
                y_pred_logits = self._forward(X_batch)
                loss_val = self.loss.forward(y_pred_logits, y_batch)
                epoch_loss += loss_val * X_batch.shape[0]
                loss_grad = self.loss.backward()
                self._backward(loss_grad)
                self.optimizer.step()
            avg_epoch_loss = epoch_loss / n_samples
            history['loss'].append(avg_epoch_loss)
            log_msg = f"Epoch {epoch+1}/{epochs} - loss: {avg_epoch_loss:.4f}"
            if validation_data:
                X_val, y_val = validation_data
                val_logits = self._forward(X_val)
                val_loss = self.loss.forward(val_logits, y_val)
                val_preds = self.predict(X_val)
                val_acc = np.mean(val_preds == y_val)
                history['val_loss'].append(val_loss)
                history['val_acc'].append(val_acc)
                log_msg += f" - val_loss: {val_loss:.4f} - val_acc: {val_acc:.4f}"
            print(log_msg)
        return history

    def predict_proba(self, X):
        logits = self._forward(X)
        logits_stable = logits - np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(logits_stable)
        probas = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        return probas

    def predict(self, X):
        probas = self.predict_proba(X)
        return np.argmax(probas, axis=1)
