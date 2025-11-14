# my_ml_project/pyml/svm/_kernels.py

import numpy as np

def linear_kernel(x1, x2):
    return np.dot(x1, x2.T)

def polynomial_kernel(x1, x2, p=3, coef0=1):
    return (np.dot(x1, x2.T) + coef0)**p

def rbf_kernel(x1, x2, gamma=0.1):
    x1_norm = np.sum(x1**2, axis=-1)
    x2_norm = np.sum(x2**2, axis=-1)
    K = np.exp(-gamma * (x1_norm[:, np.newaxis] + x2_norm[np.newaxis, :] - 2 * np.dot(x1, x2.T)))
    return K
