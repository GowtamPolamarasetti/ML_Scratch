from ._classification import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score
)
from ._regression import mean_squared_error, r2_score

__all__ = [
    'accuracy_score',
    'precision_score',
    'recall_score',
    'f1_score',
    'mean_squared_error',
    'r2_score'
]