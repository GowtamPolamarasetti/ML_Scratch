# pyml

A comprehensive Machine Learning library built from scratch in Python.

## Overview

`pyml` is a custom-built machine learning library designed to provide a clear and understandable implementation of core ML algorithms. It mimics the API of popular libraries like scikit-learn, making it easy to use for those familiar with the Python ML ecosystem.

## Features

The library includes implementations of various machine learning components:

*   **Clustering**: K-Means, etc.
*   **Decomposition**: PCA, etc.
*   **Ensemble**: Random Forests, Gradient Boosting, etc.
*   **Linear Models**: Linear Regression, Logistic Regression, etc.
*   **Neural Networks**:
    *   Sequential models
    *   Dense layers
    *   Activation functions (ReLU, Sigmoid, Tanh, etc.)
    *   Loss functions (MSE, CrossEntropy, etc.)
    *   Optimizers (SGD, Adam, etc.)
*   **Preprocessing**: Scalers, Encoders, etc.
*   **Model Selection**: Train/Test split, Cross-validation.
*   **Metrics**: Accuracy, Precision, Recall, F1-score, etc.
*   **Naive Bayes**: Gaussian, Multinomial.
*   **Neighbors**: K-Nearest Neighbors.
*   **SVM**: Support Vector Machines.
*   **Trees**: Decision Trees.

## Installation

To install the package and its dependencies, you can use `pip`:

```bash
pip install .
```

Or to install dependencies listed in `pyproject.toml` or `requirements.txt`:

```bash
pip install -r requirements.txt
```

## Usage

`pyml` follows a scikit-learn style API with `fit`, `predict`, and `transform` methods.

### Example: Neural Network on MNIST

Here is an example of how to build and train a simple Neural Network using `pyml` (based on `examples/01.py`):

```python
import numpy as np
from pyml.neural_network import Sequential, Dense, ActivationLayer
from pyml.neural_network._activations import ReLU
from pyml.neural_network._losses import SoftmaxCrossEntropyLoss
from pyml.neural_network._optimizers import Adam
from pyml.metrics import accuracy_score

# Assume X_train, y_train, X_test, y_test are loaded and preprocessed

# 1. Build Model
model = Sequential()
model.add(Dense(n_input=784, n_output=128))
model.add(ActivationLayer(ReLU()))
model.add(Dense(n_input=128, n_output=64))
model.add(ActivationLayer(ReLU()))
model.add(Dense(n_input=64, n_output=10))

# 2. Compile Model
model.compile(
    loss=SoftmaxCrossEntropyLoss(),
    optimizer_class=Adam,
    learning_rate=0.001
)

# 3. Train Model
model.fit(
    X_train, y_train, 
    epochs=10, 
    batch_size=64,
    validation_data=(X_test, y_test)
)

# 4. Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy * 100:.2f}%")
```

For more examples, check the `examples/` directory.

## Testing

The project uses `pytest` for testing. To run the tests:

```bash
pytest
```

## Authors

*   **Gowtam Polamarasetti**
