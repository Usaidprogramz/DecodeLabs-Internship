"""
feature_engineering.py
------------------------
Engineers the fraud-specific features on top of Project 1's already
cleaned dataset, and assembles the final model-ready feature matrix.

Leakage guardrails (read this before changing anything here):
---------------------------------------------------------------
- `OrderStatus` is the column IsFraud is derived from. It must NEVER
  appear in the feature matrix X -- including it would let the model
  "predict" the target by reading the target. `drop_leaky_columns()`
  removes it explicitly, and a unit test asserts it's gone.
- `WeekendFraud` (Quantity x IsFraud interaction, as specified in the
  brief) is built ONLY for exploratory charts. It is deliberately
  excluded from X for the same reason -- it's a function of the label
  itself.
- Identifier columns (OrderID, CustomerID, TrackingNumber,
  ShippingAddress, raw Date, raw CouponCode) carry no generalizable
  signal and are dropped before modeling.
"""

import numpy as np
import pandas as pd

# Columns that must never enter the feature matrix, and why.
LEAKY_OR_ID_COLUMNS = [
    "OrderStatus",      # target source
    "WeekendFraud",      # function of the target -- exploratory only
    "OrderID",           # identifier
    "CustomerID",         # identifier
    "TrackingNumber",     # identifier
    "ShippingAddress",    # identifier, near-unique per row
    "Date",               # already decomposed into OrderMonth/OrderDayOfWeek/IsWeekendOrder
    "CouponCode",          # already encoded into CouponUsed
]

CATEGORICAL_COLS = ["Product", "PaymentMethod", "ReferralSource"]


def engineer_fraud_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add the fraud-track features specified in the brief:
      - AvgItemValue  = TotalPrice / ItemsInCart
      - ItemsPerOrder = ItemsInCart / (Quantity + 1)
      - IsHighValue   = TotalPrice > 95th percentile of TotalPrice
      - PricePerUnit  = TotalPrice / Quantity
      - HasDiscount   = CouponUsed (already exists in Project 1 output; aliased here)
      - WeekendFraud  = IsWeekendOrder * IsFraud  (exploratory only -- see module docstring)

    `IsFraud` must already exist (run target_builder.build_fraud_target first).
    """
    df = df.copy()

    df["AvgItemValue"] = (df["TotalPrice"] / df["ItemsInCart"]).round(3)
    df["ItemsPerOrder"] = (df["ItemsInCart"] / (df["Quantity"] + 1)).round(3)

    high_value_threshold = df["TotalPrice"].quantile(0.95)
    df["IsHighValue"] = (df["TotalPrice"] > high_value_threshold).astype(int)

    df["PricePerUnit"] = (df["TotalPrice"] / df["Quantity"]).round(3)
    df["HasDiscount"] = df["CouponUsed"]

    # Exploratory-only interaction term -- see module docstring. Never
    # goes into the feature matrix.
    df["WeekendFraud"] = df["IsWeekendOrder"] * df["IsFraud"]

    return df


def drop_leaky_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove target-derived and identifier columns before modeling."""
    cols_to_drop = [c for c in LEAKY_OR_ID_COLUMNS if c in df.columns]
    return df.drop(columns=cols_to_drop)


def build_feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Build the final (X, y) used for modeling:
      1. drop leaky/ID columns
      2. one-hot encode categorical columns
      3. split out y = IsFraud

    Returns (X, y) with X fully numeric.
    """
    df = drop_leaky_columns(df)

    y = df["IsFraud"]
    X = df.drop(columns=["IsFraud"])

    present_categoricals = [c for c in CATEGORICAL_COLS if c in X.columns]
    X = pd.get_dummies(X, columns=present_categoricals, drop_first=False)

    # Ensure everything is numeric (guards against a stray object column
    # sneaking through and silently breaking sklearn downstream).
    non_numeric = X.select_dtypes(exclude=[np.number, bool]).columns.tolist()
    if non_numeric:
        raise ValueError(f"Non-numeric columns leaked into feature matrix: {non_numeric}")

    X = X.astype(float)

    return X, y
