"""Response models for walk-forward validation results."""

from typing import List, Optional

from pydantic import BaseModel


class FoldResult(BaseModel):
    fold: int
    train_size: int
    test_size: int
    accuracy: float
    train_start: str
    train_end: str
    test_start: str
    test_end: str


class ConfusionMatrixResult(BaseModel):
    true_negatives: int
    false_positives: int
    false_negatives: int
    true_positives: int


class WalkForwardResult(BaseModel):
    model_type: str
    n_splits: int
    overall_accuracy: float
    overall_precision: Optional[float] = None
    overall_recall: Optional[float] = None
    overall_f1: Optional[float] = None
    overall_roc_auc: Optional[float] = None
    confusion_matrix: Optional[ConfusionMatrixResult] = None
    folds: List[FoldResult]
