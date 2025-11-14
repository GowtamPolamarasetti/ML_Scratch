# my_ml_project/pyml/metrics/_classification.py

import numpy as np

def accuracy_score(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return np.mean(y_true == y_pred)

def _check_binary(y_true, y_pred):
    labels = np.unique(np.concatenate((y_true, y_pred)))
    if len(labels) > 2:
        raise ValueError("precision_score, recall_score, and f1_score are currently only supported for binary classification.")
    pos_label = np.max(labels)
    if pos_label == 0 and 0 in labels:
        pos_label = 0
    return y_true, y_pred, pos_label

def precision_score(y_true, y_pred, zero_division=0.0):
    y_true, y_pred, pos_label = _check_binary(y_true, y_pred)
    tp = np.sum((y_true == pos_label) & (y_pred == pos_label))
    fp = np.sum((y_true != pos_label) & (y_pred == pos_label))
    denominator = tp + fp
    if denominator == 0:
        return zero_division
    return tp / denominator

def recall_score(y_true, y_pred, zero_division=0.0):
    y_true, y_pred, pos_label = _check_binary(y_true, y_pred)
    tp = np.sum((y_true == pos_label) & (y_pred == pos_label))
    fn = np.sum((y_true == pos_label) & (y_pred != pos_label))
    denominator = tp + fn
    if denominator == 0:
        return zero_division
    return tp / denominator

def f1_score(y_true, y_pred, zero_division=0.0):
    p = precision_score(y_true, y_pred, zero_division=zero_division)
    r = recall_score(y_true, y_pred, zero_division=zero_division)
    denominator = p + r
    if denominator == 0:
        return zero_division
    return 2 * (p * r) / denominator
