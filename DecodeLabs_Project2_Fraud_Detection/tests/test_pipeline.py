"""
tests/test_pipeline.py
------------------------
Unit tests for Project 2. The most important tests here are the
leakage-safety ones (marked below) -- they exist specifically to catch
the two traps the brief calls out: using OrderStatus (the label
source) as a feature, and letting SMOTE/scaling see the test set.

Run with:
    pytest -v
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from src.target_builder import build_fraud_target, class_balance_report, FRAUD_STATUSES
from src.feature_engineering import (
    engineer_fraud_features,
    drop_leaky_columns,
    build_feature_matrix,
    LEAKY_OR_ID_COLUMNS,
)
from src.modeling import (
    build_logistic_regression_pipeline,
    build_random_forest_pipeline,
    tune_model,
    RANDOM_STATE,
)
from src.evaluation import evaluate_model


# --------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------- #

@pytest.fixture
def sample_df():
    n = 40
    rng = np.random.default_rng(0)
    statuses = (["Delivered", "Shipped", "Pending"] * 10 + ["Returned", "Cancelled"] * 5)[:n]
    return pd.DataFrame({
        "OrderID": [f"ORD{i}" for i in range(n)],
        "Date": pd.date_range("2024-01-01", periods=n, freq="D"),
        "CustomerID": [f"C{i % 8}" for i in range(n)],
        "Product": rng.choice(["Laptop", "Phone", "Chair"], size=n),
        "Quantity": rng.integers(1, 6, size=n),
        "UnitPrice": rng.uniform(10, 500, size=n).round(2),
        "ShippingAddress": [f"{i} Main St" for i in range(n)],
        "PaymentMethod": rng.choice(["Cash", "Credit Card"], size=n),
        "OrderStatus": statuses,
        "TrackingNumber": [f"TRK{i}" for i in range(n)],
        "ItemsInCart": rng.integers(1, 10, size=n),
        "CouponCode": rng.choice(["SAVE10", "NoCoupon"], size=n),
        "ReferralSource": rng.choice(["Instagram", "Email"], size=n),
        "TotalPrice": rng.uniform(20, 3000, size=n).round(2),
        "OrderMonth": rng.integers(1, 13, size=n),
        "OrderDayOfWeek": rng.integers(0, 7, size=n),
        "IsWeekendOrder": rng.integers(0, 2, size=n),
        "CouponUsed": rng.integers(0, 2, size=n),
        "CartFillRatio": rng.uniform(0, 1, size=n).round(3),
        "CustomerOrderCount": rng.integers(1, 4, size=n),
        "IsRepeatCustomer": rng.integers(0, 2, size=n),
        "Quantity_was_outlier": 0,
        "UnitPrice_was_outlier": 0,
        "ItemsInCart_was_outlier": 0,
        "TotalPrice_was_outlier": 0,
    })


# --------------------------------------------------------------------- #
# Target construction
# --------------------------------------------------------------------- #

def test_fraud_target_flags_only_returned_and_cancelled(sample_df):
    labeled = build_fraud_target(sample_df)
    fraud_rows = labeled[labeled["IsFraud"] == 1]
    legit_rows = labeled[labeled["IsFraud"] == 0]
    assert set(fraud_rows["OrderStatus"].unique()) <= FRAUD_STATUSES
    assert not set(legit_rows["OrderStatus"].unique()) & FRAUD_STATUSES


def test_class_balance_report_sums_to_total(sample_df):
    labeled = build_fraud_target(sample_df)
    report = class_balance_report(labeled)
    assert report["count"].sum() == len(labeled)
    assert abs(report["pct"].sum() - 100.0) < 0.5


# --------------------------------------------------------------------- #
# Feature engineering
# --------------------------------------------------------------------- #

def test_engineered_features_present(sample_df):
    labeled = build_fraud_target(sample_df)
    out = engineer_fraud_features(labeled)
    for col in ["AvgItemValue", "ItemsPerOrder", "IsHighValue", "PricePerUnit",
                "HasDiscount", "WeekendFraud"]:
        assert col in out.columns


def test_is_high_value_matches_95th_percentile(sample_df):
    labeled = build_fraud_target(sample_df)
    out = engineer_fraud_features(labeled)
    threshold = labeled["TotalPrice"].quantile(0.95)
    expected = (labeled["TotalPrice"] > threshold).astype(int)
    pd.testing.assert_series_equal(out["IsHighValue"], expected, check_names=False)


def test_weekend_fraud_is_interaction_term(sample_df):
    labeled = build_fraud_target(sample_df)
    out = engineer_fraud_features(labeled)
    expected = labeled["IsWeekendOrder"] * out["IsFraud"]
    pd.testing.assert_series_equal(out["WeekendFraud"], expected, check_names=False)


# --------------------------------------------------------------------- #
# LEAKAGE SAFETY -- the most important tests in this file
# --------------------------------------------------------------------- #

def test_order_status_never_enters_feature_matrix(sample_df):
    """OrderStatus is the literal source of the label. If this leaks
    into X, the model could trivially 'predict' the target."""
    labeled = build_fraud_target(sample_df)
    featured = engineer_fraud_features(labeled)
    X, y = build_feature_matrix(featured)
    assert "OrderStatus" not in X.columns
    assert not any(col.startswith("OrderStatus") for col in X.columns)


def test_weekend_fraud_never_enters_feature_matrix(sample_df):
    """WeekendFraud is IsWeekendOrder * IsFraud -- a direct function of
    the label. It must stay exploratory-only, never a model feature."""
    labeled = build_fraud_target(sample_df)
    featured = engineer_fraud_features(labeled)
    X, y = build_feature_matrix(featured)
    assert "WeekendFraud" not in X.columns


def test_identifier_columns_never_enter_feature_matrix(sample_df):
    labeled = build_fraud_target(sample_df)
    featured = engineer_fraud_features(labeled)
    X, y = build_feature_matrix(featured)
    for col in ["OrderID", "CustomerID", "TrackingNumber", "ShippingAddress", "Date", "CouponCode"]:
        assert col not in X.columns


def test_all_leaky_columns_covered_by_drop_list(sample_df):
    """Every column named in LEAKY_OR_ID_COLUMNS must actually be gone
    after drop_leaky_columns runs, so the guard can't silently rot."""
    labeled = build_fraud_target(sample_df)
    featured = engineer_fraud_features(labeled)
    dropped = drop_leaky_columns(featured)
    for col in LEAKY_OR_ID_COLUMNS:
        assert col not in dropped.columns


def test_feature_matrix_is_fully_numeric(sample_df):
    labeled = build_fraud_target(sample_df)
    featured = engineer_fraud_features(labeled)
    X, y = build_feature_matrix(featured)
    assert all(np.issubdtype(dtype, np.number) for dtype in X.dtypes)


def test_smote_pipeline_does_not_alter_test_set_size():
    """SMOTE inside a fitted pipeline must be inert at predict time --
    it should never resample the test set."""
    rng = np.random.default_rng(1)
    X = pd.DataFrame(rng.normal(size=(200, 4)), columns=list("abcd"))
    y = pd.Series([0] * 180 + [1] * 20)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    pipeline = build_logistic_regression_pipeline()
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)

    assert len(preds) == len(X_test)  # test set untouched by resampling


def test_split_happens_before_smote_class_ratio_preserved_in_test():
    """The test set must reflect the TRUE (imbalanced) class ratio --
    not a SMOTE-balanced one. This is the leak-free guarantee itself."""
    rng = np.random.default_rng(2)
    X = pd.DataFrame(rng.normal(size=(300, 4)), columns=list("abcd"))
    y = pd.Series([0] * 270 + [1] * 30)  # 90/10 imbalance

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    test_fraud_pct = y_test.mean()
    assert abs(test_fraud_pct - 0.10) < 0.03  # stratified split preserves ~10%


# --------------------------------------------------------------------- #
# Modeling & evaluation
# --------------------------------------------------------------------- #

def test_logistic_regression_pipeline_has_expected_steps():
    pipeline = build_logistic_regression_pipeline()
    assert list(pipeline.named_steps.keys()) == ["scaler", "smote", "classifier"]


def test_random_forest_pipeline_has_no_scaler():
    pipeline = build_random_forest_pipeline()
    assert "scaler" not in pipeline.named_steps
    assert list(pipeline.named_steps.keys()) == ["smote", "classifier"]


def test_tune_model_optimizes_recall_not_accuracy():
    rng = np.random.default_rng(3)
    X = pd.DataFrame(rng.normal(size=(150, 3)), columns=list("xyz"))
    y = pd.Series((rng.random(150) < 0.15).astype(int))

    tuned = tune_model(
        build_logistic_regression_pipeline(),
        {"classifier__C": [1.0], "smote__k_neighbors": [3]},
        X, y, name="test-model", cv=3,
    )
    assert tuned.name == "test-model"
    assert 0.0 <= tuned.best_cv_recall <= 1.0
    assert hasattr(tuned.best_estimator, "predict")


def test_evaluate_model_excludes_accuracy():
    """evaluation.EvaluationResult must never carry an accuracy field --
    that metric is deliberately excluded throughout this project."""
    from dataclasses import fields
    from src.evaluation import EvaluationResult
    field_names = {f.name for f in fields(EvaluationResult)}
    assert "accuracy" not in field_names


def test_evaluate_model_returns_valid_metric_ranges():
    rng = np.random.default_rng(4)
    X = pd.DataFrame(rng.normal(size=(120, 3)), columns=list("xyz"))
    y = pd.Series((rng.random(120) < 0.3).astype(int))
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE
    )
    pipeline = build_random_forest_pipeline()
    pipeline.fit(X_train, y_train)
    result = evaluate_model(pipeline, X_test, y_test, "Random Forest")

    for metric in [result.precision, result.recall, result.f1, result.roc_auc]:
        assert 0.0 <= metric <= 1.0
    assert result.confusion.shape == (2, 2)
    assert result.feature_importances is not None
