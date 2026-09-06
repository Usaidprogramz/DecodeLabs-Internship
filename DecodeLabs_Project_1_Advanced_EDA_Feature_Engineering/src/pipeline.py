"""
pipeline.py
------------
Orchestrates the full Input -> Process -> Output pipeline:

    PHASE 1  Input      : load raw data, handle missing values, detect
                           and neutralize outliers.
    PHASE 2  Process     : engineer new predictive features, one-hot
                           encode categoricals, check/resolve
                           multicollinearity.
    PHASE 3  Output      : validate the final dataset against a schema
                           contract, then persist the clean artifacts.

Run directly with:
    python -m src.pipeline
or via the project's main.py.
"""

from pathlib import Path

import pandas as pd

from src.data_loader import load_raw_dataset, dataset_overview
from src.missing_value_handler import handle_missing_values
from src.outlier_handler import detect_outliers, winsorize_columns, outlier_report_to_dataframe
from src.feature_engineering import engineer_features, one_hot_encode, find_multicollinear_pairs
from src.schema_validator import ColumnRule, validate_schema
from src.scaler import scale_numeric_features

RAW_PATH = Path("data/raw/Dataset_for_Data_Analytics.xlsx")
PROCESSED_DIR = Path("data/processed")
REPORTS_DIR = Path("reports")

NUMERIC_COLS = ["Quantity", "UnitPrice", "ItemsInCart", "TotalPrice"]
CATEGORICAL_COLS = ["CouponCode"]
ENCODE_COLS = ["Product", "PaymentMethod", "OrderStatus", "ReferralSource", "CouponCode"]

# Continuous numeric features scaled to zero mean / unit variance for the
# final model-ready output. Binary flags (IsWeekendOrder, CouponUsed,
# IsRepeatCustomer, every *_was_outlier column) and one-hot encoded
# columns are deliberately excluded -- scaling a 0/1 indicator doesn't
# help most estimators and only makes the column harder to read.
SCALE_COLS = [
    "Quantity", "UnitPrice", "ItemsInCart", "TotalPrice",
    "CartFillRatio", "CustomerOrderCount", "OrderMonth", "OrderDayOfWeek",
]


def run_pipeline(raw_path: Path = RAW_PATH, save_outputs: bool = True) -> dict:
    """Execute the full pipeline and return every intermediate artifact."""

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------- PHASE 1
    df_raw = load_raw_dataset(raw_path)
    overview_before = dataset_overview(df_raw)

    # NUMERIC_COLS have 0% missingness in this dataset (verified during
    # EDA), so handle_missing_values is a no-op for them here -- it is
    # still wired in so the pipeline is robust to future data refreshes
    # that do contain numeric gaps.
    df_clean, imputation_log = handle_missing_values(
        df_raw,
        numeric_cols=NUMERIC_COLS,
        categorical_cols=[],
    )

    # CouponCode is the one column with real missingness (25.75%),
    # which the generic decision matrix would route to KNN imputation
    # (>20% band). We deliberately override that: a missing coupon
    # code almost certainly means "no coupon was applied at checkout"
    # (Missing Not At Random), not a random data-entry gap. The
    # CouponUsed flag below captures that signal from the ORIGINAL
    # null pattern; only after that do we fill CouponCode itself with
    # a domain-meaningful sentinel instead of a statistical guess.
    coupon_pct_missing = df_clean["CouponCode"].isna().mean() * 100

    df_features = engineer_features(df_clean)

    outlier_reports = detect_outliers(df_features, NUMERIC_COLS)
    df_capped = winsorize_columns(df_features, NUMERIC_COLS, add_flag_columns=True)

    df_capped["CouponCode"] = df_capped["CouponCode"].fillna("NoCoupon")
    imputation_log.add(
        "CouponCode", coupon_pct_missing, "domain_sentinel_override",
        "25.75% missing falls in the >20% KNN band, but missingness here is "
        "MNAR (no coupon applied), so filled with sentinel 'NoCoupon' instead "
        "of statistical estimation.",
    )

    # ---------------------------------------------------------- PHASE 2
    collinear_pairs = find_multicollinear_pairs(
        df_capped, NUMERIC_COLS + ["CartFillRatio", "CustomerOrderCount"]
    )

    df_encoded = one_hot_encode(df_capped, ENCODE_COLS)

    # Standardize continuous numeric features (zero mean / unit variance)
    # for the final model-ready output. This runs AFTER winsorization so
    # the scaler's mean/std reflect the outlier-safe distribution, not
    # the raw one. A separate scaled copy is kept alongside the unscaled
    # encoded dataset so the human-readable version stays interpretable.
    df_scaled, fitted_scaler, scaling_log = scale_numeric_features(df_encoded, SCALE_COLS)

    # ---------------------------------------------------------- PHASE 3
    rules = [
        ColumnRule("OrderID", "string", nullable=False),
        ColumnRule("Date", "datetime", nullable=False),
        ColumnRule("Quantity", "numeric", min_value=0, nullable=False),
        ColumnRule("UnitPrice", "numeric", min_value=0, nullable=False),
        ColumnRule("TotalPrice", "numeric", min_value=0, nullable=False),
        ColumnRule("ItemsInCart", "numeric", min_value=0, nullable=False),
        ColumnRule("CouponUsed", "numeric", min_value=0, max_value=1, nullable=False),
        ColumnRule("IsWeekendOrder", "numeric", min_value=0, max_value=1, nullable=False),
        ColumnRule("CartFillRatio", "numeric", min_value=0, max_value=1, nullable=False),
    ]
    validation_result = validate_schema(df_capped, rules)

    if save_outputs:
        df_capped.to_csv(PROCESSED_DIR / "cleaned_dataset.csv", index=False)
        df_encoded.to_csv(PROCESSED_DIR / "ml_ready_encoded_dataset.csv", index=False)
        df_scaled.to_csv(PROCESSED_DIR / "ml_ready_scaled_dataset.csv", index=False)
        overview_before.to_csv(REPORTS_DIR / "raw_data_overview.csv")
        imputation_log.as_dataframe().to_csv(REPORTS_DIR / "imputation_log.csv", index=False)
        outlier_report_to_dataframe(outlier_reports).to_csv(REPORTS_DIR / "outlier_report.csv", index=False)
        collinear_pairs.to_csv(REPORTS_DIR / "multicollinearity_report.csv", index=False)
        validation_result.as_dataframe().to_csv(REPORTS_DIR / "schema_validation_failures.csv", index=False)
        scaling_log.as_dataframe().to_csv(REPORTS_DIR / "scaling_log.csv", index=False)

    return {
        "raw": df_raw,
        "overview_before": overview_before,
        "cleaned": df_capped,
        "encoded": df_encoded,
        "scaled": df_scaled,
        "scaling_log": scaling_log,
        "imputation_log": imputation_log,
        "outlier_reports": outlier_reports,
        "collinear_pairs": collinear_pairs,
        "validation_result": validation_result,
    }


if __name__ == "__main__":
    results = run_pipeline()
    print(f"Rows in / out: {len(results['raw'])} -> {len(results['cleaned'])}")
    print(f"New feature columns added: {len(results['cleaned'].columns) - len(results['raw'].columns)}")
    print(f"Schema validation passed: {results['validation_result'].passed}")
    print("Clean dataset written to data/processed/cleaned_dataset.csv")
    print("ML-ready encoded dataset written to data/processed/ml_ready_encoded_dataset.csv")
    print("ML-ready scaled dataset written to data/processed/ml_ready_scaled_dataset.csv")
