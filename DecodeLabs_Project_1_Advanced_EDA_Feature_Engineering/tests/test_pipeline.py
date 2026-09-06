"""
tests/test_pipeline.py
------------------------
Unit tests for each stage of the Input -> Process -> Output pipeline.

Run with:
    pytest -v
"""

import numpy as np
import pandas as pd
import pytest

from src.missing_value_handler import handle_missing_values
from src.outlier_handler import compute_iqr_bounds, detect_outliers, winsorize_columns
from src.feature_engineering import engineer_features, one_hot_encode, find_multicollinear_pairs
from src.schema_validator import ColumnRule, validate_schema
from src.scaler import scale_numeric_features


# --------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------- #

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "Date": pd.to_datetime(
            ["2024-01-01", "2024-01-08", "2024-01-15", "2024-01-22", "2024-01-29"]
        ),
        "CustomerID": ["C1", "C1", "C2", "C3", "C2"],
        "Quantity": [1, 2, 3, 4, 5],
        "UnitPrice": [10.0, 20.0, 30.0, 40.0, 5000.0],  # 5000 is a deliberate outlier
        "ItemsInCart": [2, 4, 6, 8, 10],
        "CouponCode": ["SAVE10", None, "FREESHIP", None, "SAVE10"],
        "TotalPrice": [10.0, 40.0, 90.0, 160.0, 25000.0],
    })


# --------------------------------------------------------------------- #
# Missing value handling
# --------------------------------------------------------------------- #

def test_numeric_missing_below_5pct_drops_rows():
    df = pd.DataFrame({"x": [1, 2, 3, np.nan] + list(range(4, 101))})  # ~1% missing
    cleaned, log = handle_missing_values(df, numeric_cols=["x"])
    assert cleaned["x"].isna().sum() == 0
    assert len(cleaned) == len(df) - 1
    assert log.as_dataframe().iloc[0]["strategy"] == "drop_rows"


def test_numeric_missing_5_to_20pct_uses_median():
    values = list(range(1, 101))
    values[:15] = [np.nan] * 15  # 15% missing
    df = pd.DataFrame({"x": values})
    cleaned, log = handle_missing_values(df, numeric_cols=["x"])
    assert cleaned["x"].isna().sum() == 0
    assert len(cleaned) == len(df)  # no rows dropped
    assert "global_median" in log.as_dataframe()["strategy"].values


def test_numeric_missing_above_20pct_uses_knn():
    rng = np.random.default_rng(0)
    x = rng.normal(size=100)
    y = x * 2 + rng.normal(scale=0.1, size=100)
    mask = rng.choice(100, size=30, replace=False)  # 30% missing
    x[mask] = np.nan
    df = pd.DataFrame({"x": x, "y": y})
    cleaned, log = handle_missing_values(df, numeric_cols=["x", "y"])
    assert cleaned["x"].isna().sum() == 0
    assert "knn_imputation" in log.as_dataframe()["strategy"].values


def test_categorical_missing_filled_with_sentinel(sample_df):
    cleaned, log = handle_missing_values(sample_df, categorical_cols=["CouponCode"])
    assert cleaned["CouponCode"].isna().sum() == 0
    assert (cleaned["CouponCode"] == "Unknown").sum() == 2


# --------------------------------------------------------------------- #
# Outlier detection & winsorization
# --------------------------------------------------------------------- #

def test_iqr_bounds_flag_the_planted_outlier(sample_df):
    reports = detect_outliers(sample_df, ["UnitPrice"])
    assert reports[0].n_outliers == 1


def test_winsorize_caps_without_dropping_rows(sample_df):
    capped = winsorize_columns(sample_df, ["UnitPrice"])
    assert len(capped) == len(sample_df)  # row count preserved
    q1, q3, lower, upper = compute_iqr_bounds(sample_df["UnitPrice"])
    assert capped["UnitPrice"].max() <= upper
    assert "UnitPrice_was_outlier" in capped.columns
    assert capped["UnitPrice_was_outlier"].sum() == 1


# --------------------------------------------------------------------- #
# Feature engineering
# --------------------------------------------------------------------- #

def test_engineer_features_adds_expected_columns(sample_df):
    out = engineer_features(sample_df)
    for col in ["OrderMonth", "OrderDayOfWeek", "IsWeekendOrder",
                "CouponUsed", "CartFillRatio", "CustomerOrderCount", "IsRepeatCustomer"]:
        assert col in out.columns


def test_cart_fill_ratio_is_bounded_and_correct(sample_df):
    out = engineer_features(sample_df)
    expected = (sample_df["Quantity"] / sample_df["ItemsInCart"]).round(3)
    pd.testing.assert_series_equal(out["CartFillRatio"], expected, check_names=False)


def test_repeat_customer_flagging(sample_df):
    out = engineer_features(sample_df)
    # C1 and C2 each appear twice; C3 appears once.
    c1_rows = out[out["CustomerID"] == "C1"]
    c3_rows = out[out["CustomerID"] == "C3"]
    assert (c1_rows["IsRepeatCustomer"] == 1).all()
    assert (c3_rows["IsRepeatCustomer"] == 0).all()


def test_one_hot_encoding_removes_original_column(sample_df):
    encoded = one_hot_encode(sample_df, ["CustomerID"])
    assert "CustomerID" not in encoded.columns
    assert any(c.startswith("CustomerID_") for c in encoded.columns)


def test_find_multicollinear_pairs_detects_perfect_correlation():
    df = pd.DataFrame({"a": range(50), "b": [x * 2 for x in range(50)], "c": [1, 2] * 25})
    pairs = find_multicollinear_pairs(df, ["a", "b", "c"], threshold=0.8)
    assert not pairs.empty
    assert {"a", "b"} == {pairs.iloc[0]["feature_a"], pairs.iloc[0]["feature_b"]}


# --------------------------------------------------------------------- #
# Schema validation
# --------------------------------------------------------------------- #

def test_schema_validation_passes_on_clean_data():
    df = pd.DataFrame({"Quantity": [1, 2, 3]})
    rules = [ColumnRule("Quantity", "numeric", min_value=0, nullable=False)]
    result = validate_schema(df, rules)
    assert result.passed


def test_schema_validation_flags_negative_values_and_nulls():
    df = pd.DataFrame({"Quantity": [1, -5, np.nan]})
    rules = [ColumnRule("Quantity", "numeric", min_value=0, nullable=False)]
    result = validate_schema(df, rules)
    assert not result.passed
    checks = result.as_dataframe()["check"].tolist()
    assert "nullability" in checks
    assert "min_bound" in checks


def test_schema_validation_flags_missing_column():
    df = pd.DataFrame({"Other": [1, 2, 3]})
    rules = [ColumnRule("Quantity", "numeric")]
    result = validate_schema(df, rules)
    assert not result.passed
    assert result.as_dataframe().iloc[0]["check"] == "presence"


# --------------------------------------------------------------------- #
# Feature scaling
# --------------------------------------------------------------------- #

def test_scale_numeric_features_produces_zero_mean_unit_std():
    df = pd.DataFrame({
        "x": [10.0, 20.0, 30.0, 40.0, 50.0],
        "flag": [0, 1, 0, 1, 0],  # left untouched
    })
    scaled, scaler, log = scale_numeric_features(df, ["x"])
    assert scaled["x"].mean() == pytest.approx(0.0, abs=1e-9)
    assert scaled["x"].std(ddof=0) == pytest.approx(1.0, abs=1e-9)
    # Untouched column is unchanged
    pd.testing.assert_series_equal(scaled["flag"], df["flag"])


def test_scale_numeric_features_logs_pre_scale_stats():
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})
    _, _, log = scale_numeric_features(df, ["x"])
    log_df = log.as_dataframe()
    assert log_df.iloc[0]["column"] == "x"
    assert log_df.iloc[0]["pre_scale_mean"] == pytest.approx(3.0)


def test_scaler_can_transform_new_data_consistently():
    df = pd.DataFrame({"x": [10.0, 20.0, 30.0, 40.0, 50.0]})
    _, fitted_scaler, _ = scale_numeric_features(df, ["x"])
    new_point = pd.DataFrame({"x": [30.0]})  # the training mean
    transformed = fitted_scaler.transform(new_point)
    assert transformed[0][0] == pytest.approx(0.0, abs=1e-9)
