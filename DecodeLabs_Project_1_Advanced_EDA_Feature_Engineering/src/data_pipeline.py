"""DecodeLabs Project 1 - Advanced EDA & Feature Engineering pipeline.

The implementation follows an Input -> Process -> Output design:
Input: validate and profile raw orders
Process: impute, flag/cap outliers, engineer features, encode categories
Output: cleaned feature table + validation-ready artifacts
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import OneHotEncoder

NUMERIC_COLUMNS = ["Quantity", "UnitPrice", "ItemsInCart", "TotalPrice"]
CATEGORICAL_COLUMNS = ["Product", "PaymentMethod", "OrderStatus", "ReferralSource"]


def load_data(path: str | Path) -> pd.DataFrame:
    df = pd.read_excel(path)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    return df


def missingness_profile(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({
        "missing_count": df.isna().sum(),
        "missing_pct": df.isna().mean().mul(100),
        "dtype": df.dtypes.astype(str),
        "n_unique": df.nunique(dropna=True),
    })
    return out.sort_values("missing_pct", ascending=False)


def impute_numeric(df: pd.DataFrame, strategy: str = "median") -> pd.DataFrame:
    """Reusable numeric imputation required by the brief.

    Supports mean, median, or KNN. The supplied dataset currently contains no
    missing numeric values, so this function is intentionally reusable rather
    than forcing synthetic values into clean columns.
    """
    out = df.copy()
    cols = [c for c in NUMERIC_COLUMNS if c in out.columns and out[c].isna().any()]
    if not cols:
        return out
    if strategy in {"mean", "median"}:
        fill = out[cols].mean() if strategy == "mean" else out[cols].median()
        out[cols] = out[cols].fillna(fill)
    elif strategy == "knn":
        out[cols] = KNNImputer(n_neighbors=5).fit_transform(out[cols])
    else:
        raise ValueError("strategy must be one of: mean, median, knn")
    return out


def impute_categorical(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # In this dataset a blank coupon naturally represents no coupon used.
    out["CouponCode_Imputed"] = out["CouponCode"].fillna("NO_COUPON")
    return out


def iqr_bounds(series: pd.Series) -> tuple[float, float]:
    q1, q3 = series.quantile([0.25, 0.75])
    iqr = q3 - q1
    return float(q1 - 1.5 * iqr), float(q3 + 1.5 * iqr)


def neutralize_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Flag and winsorize TotalPrice without deleting observations."""
    out = df.copy()
    lo, hi = iqr_bounds(out["TotalPrice"])
    out["TotalPrice_IQR_Outlier"] = ~out["TotalPrice"].between(lo, hi)
    out["TotalPrice_Winsorized"] = out["TotalPrice"].clip(lo, hi)
    return out


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["HasCoupon"] = out["CouponCode_Imputed"].ne("NO_COUPON").astype("int8")
    out["OrderYear"] = out["Date"].dt.year
    out["OrderMonth"] = out["Date"].dt.month
    out["OrderQuarter"] = out["Date"].dt.quarter
    out["OrderDayOfWeek"] = out["Date"].dt.day_name()
    out["IsWeekend"] = out["Date"].dt.dayofweek.ge(5).astype("int8")
    out["BasketFillRatio"] = out["Quantity"].div(out["ItemsInCart"]).replace([np.inf, -np.inf], np.nan)
    out["OrderValuePerCartItem"] = out["TotalPrice"].div(out["ItemsInCart"]).replace([np.inf, -np.inf], np.nan)
    return out


def correlation_audit(df: pd.DataFrame, threshold: float = 0.80) -> pd.DataFrame:
    corr = df.select_dtypes(include=np.number).corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    pairs = [(a, b, upper.loc[a, b]) for b in upper.columns for a in upper.index if pd.notna(upper.loc[a, b]) and upper.loc[a, b] > threshold]
    return pd.DataFrame(pairs, columns=["feature_a", "feature_b", "abs_correlation"]).sort_values("abs_correlation", ascending=False)


def encode_nominal_features(df: pd.DataFrame) -> tuple[pd.DataFrame, OneHotEncoder]:
    """One-hot encode nominal categories; avoids false ordinal geometry."""
    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.int8)
    matrix = enc.fit_transform(df[CATEGORICAL_COLUMNS])
    names = enc.get_feature_names_out(CATEGORICAL_COLUMNS)
    encoded = pd.DataFrame(matrix, columns=names, index=df.index)
    return pd.concat([df, encoded], axis=1), enc


def validate_business_rules(df: pd.DataFrame) -> None:
    assert df["OrderID"].is_unique, "OrderID must be unique"
    assert df["Quantity"].between(1, 5).all(), "Quantity outside expected range"
    assert df["ItemsInCart"].gt(0).all(), "ItemsInCart must be positive"
    assert df["UnitPrice"].gt(0).all(), "UnitPrice must be positive"
    assert df["TotalPrice"].gt(0).all(), "TotalPrice must be positive"
    expected = df["Quantity"] * df["UnitPrice"]
    assert np.allclose(expected, df["TotalPrice"], atol=0.01), "TotalPrice != Quantity * UnitPrice"


def run_pipeline(input_path: str | Path, output_path: str | Path) -> pd.DataFrame:
    df = load_data(input_path)
    validate_business_rules(df)
    df = impute_numeric(df, strategy="median")
    df = impute_categorical(df)
    df = neutralize_outliers(df)
    df = engineer_features(df)
    validate_business_rules(df)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return df
