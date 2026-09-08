"""
modeling.py
------------
Builds the two leak-free classification pipelines the brief requires
(Logistic Regression and Random Forest, both with SMOTE) and tunes
each with GridSearchCV, optimizing for recall rather than accuracy.

Why `imblearn.pipeline.Pipeline` instead of `sklearn.pipeline.Pipeline`:
--------------------------------------------------------------------------
A plain sklearn Pipeline expects every step to implement `.transform()`,
which only reshapes X. SMOTE needs to change BOTH X and y (it creates
new synthetic rows with an implied fraud label), so it doesn't fit that
interface -- imblearn's Pipeline understands `fit_resample()` and, by
construction, only ever applies it to the training fold, never to
validation or test data. This is what makes GridSearchCV's internal
cross-validation leak-free automatically: on every fold, SMOTE is
re-fit on that fold's training rows only.

Why the train/test split happens BEFORE any of this runs:
------------------------------------------------------------
See `run_pipeline()` in pipeline.py. SMOTE and StandardScaler must
never see the test set, or the evaluation numbers would be optimistic
and untrustworthy -- the classic "trap #2" the brief calls out.
"""

from dataclasses import dataclass

import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42


@dataclass
class TunedModel:
    name: str
    best_estimator: object
    best_params: dict
    best_cv_recall: float


def build_logistic_regression_pipeline() -> ImbPipeline:
    """StandardScaler -> SMOTE -> LogisticRegression.

    Scaling comes first because Logistic Regression's regularization
    penalty is distorted by unscaled features with very different
    ranges (e.g. TotalPrice in the hundreds vs. a 0/1 flag).
    """
    return ImbPipeline([
        ("scaler", StandardScaler()),
        ("smote", SMOTE(random_state=RANDOM_STATE)),
        ("classifier", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
    ])


def build_random_forest_pipeline() -> ImbPipeline:
    """SMOTE -> RandomForestClassifier.

    No scaler: tree-based models split on ordinal feature thresholds,
    so scale is mathematically irrelevant to a Random Forest.
    """
    return ImbPipeline([
        ("smote", SMOTE(random_state=RANDOM_STATE)),
        ("classifier", RandomForestClassifier(random_state=RANDOM_STATE)),
    ])


LOGISTIC_REGRESSION_PARAM_GRID = {
    "classifier__C": [0.01, 0.1, 1.0, 10.0],
    "classifier__solver": ["liblinear", "lbfgs"],
    "smote__k_neighbors": [3, 5, 7],
}

RANDOM_FOREST_PARAM_GRID = {
    "classifier__n_estimators": [100, 200],
    "classifier__max_depth": [None, 10, 20],
    "classifier__class_weight": [None, "balanced"],
    "smote__k_neighbors": [3, 5],
}


def tune_model(
    pipeline: ImbPipeline,
    param_grid: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    name: str,
    cv: int = 5,
) -> TunedModel:
    """
    Run GridSearchCV optimizing for recall (not accuracy -- see brief).
    Because `pipeline` is an imblearn Pipeline, SMOTE is safely re-fit
    inside every one of the `cv` folds, on that fold's training split
    only. The held-out fold in each split never sees synthetic data.
    """
    search = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        scoring="recall",
        cv=cv,
        n_jobs=-1,
    )
    search.fit(X_train, y_train)

    return TunedModel(
        name=name,
        best_estimator=search.best_estimator_,
        best_params=search.best_params_,
        best_cv_recall=search.best_score_,
    )
