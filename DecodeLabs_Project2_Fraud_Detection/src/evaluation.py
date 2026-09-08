"""
evaluation.py
---------------
Evaluates tuned models on the held-out test set using Precision,
Recall, F1, and ROC-AUC. Accuracy is deliberately never computed or
reported here -- on an imbalanced target, a model that always
predicts "legitimate" would score deceptively well on accuracy while
catching zero fraud, which is exactly the trap the brief warns about.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


@dataclass
class EvaluationResult:
    model_name: str
    precision: float
    recall: float
    f1: float
    roc_auc: float
    confusion: np.ndarray
    fpr: np.ndarray = field(repr=False)
    tpr: np.ndarray = field(repr=False)
    pr_precision: np.ndarray = field(repr=False)
    pr_recall: np.ndarray = field(repr=False)
    feature_importances: pd.Series | None = field(default=None, repr=False)


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series, name: str) -> EvaluationResult:
    """
    Score a fitted model on the untouched test set. `model` must be a
    fitted estimator/pipeline (the SMOTE step, if present, is inert at
    predict time -- it only resamples during .fit()).
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    pr_precision, pr_recall, _ = precision_recall_curve(y_test, y_proba)

    feature_importances = None
    classifier_step = None
    if hasattr(model, "named_steps") and "classifier" in model.named_steps:
        classifier_step = model.named_steps["classifier"]
    if classifier_step is not None and hasattr(classifier_step, "feature_importances_"):
        feature_importances = pd.Series(
            classifier_step.feature_importances_, index=X_test.columns
        ).sort_values(ascending=False)

    return EvaluationResult(
        model_name=name,
        precision=precision,
        recall=recall,
        f1=f1,
        roc_auc=roc_auc,
        confusion=cm,
        fpr=fpr,
        tpr=tpr,
        pr_precision=pr_precision,
        pr_recall=pr_recall,
        feature_importances=feature_importances,
    )


def results_to_dataframe(results: list[EvaluationResult]) -> pd.DataFrame:
    return pd.DataFrame([{
        "model": r.model_name,
        "precision": round(r.precision, 3),
        "recall": round(r.recall, 3),
        "f1": round(r.f1, 3),
        "roc_auc": round(r.roc_auc, 3),
    } for r in results])
