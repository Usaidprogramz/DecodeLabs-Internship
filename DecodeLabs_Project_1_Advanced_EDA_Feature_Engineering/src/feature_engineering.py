"""
feature_engineering.py
------------------------
Engineers new predictive features from the cleaned orders dataset, and
handles categorical -> coordinate-space translation (one-hot encoding)
plus a multicollinearity check on the resulting numeric feature matrix.
"""

import numpy as np
import pandas as pd


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add new predictive features to the dataframe. At least 3 new
    features are required by the brief; this creates seven, each
    documented below.

    1. OrderMonth / OrderDayOfWeek / IsWeekendOrder
       Temporal signal extracted from the Date column. Retail demand is
       strongly seasonal and day-of-week dependent, so these are high
       value predictors for downstream models.

    2. CouponUsed
       Binary flag derived from the CouponCode missingness pattern
       (missing == no coupon applied at checkout). Captures the
       missingness itself as a feature rather than discarding the
       signal during imputation.

    3. CartFillRatio
       Quantity / ItemsInCart. Measures what share of the shopping
       cart this particular order line represents -- a proxy for
       purchase decisiveness / cart abandonment risk.

    4. IsRepeatCustomer / CustomerOrderCount
       Frequency-encodes the CustomerID column. Repeat customers are
       known to behave differently (higher lifetime value, different
       return rates) than first-time buyers.
    """
    df = df.copy()

    # 1. Temporal features
    df["OrderMonth"] = df["Date"].dt.month
    df["OrderDayOfWeek"] = df["Date"].dt.dayofweek  # 0=Mon ... 6=Sun
    df["IsWeekendOrder"] = df["OrderDayOfWeek"].isin([5, 6]).astype(int)

    # 2. Coupon usage flag (computed BEFORE the missing CouponCode
    #    values are imputed with a sentinel, using the original
    #    dataframe's null pattern which the pipeline preserves for us)
    if "CouponCode" in df.columns:
        df["CouponUsed"] = df["CouponCode"].notna().astype(int)

    # 3. Cart fill ratio
    df["CartFillRatio"] = (df["Quantity"] / df["ItemsInCart"]).round(3)

    # 4. Customer frequency features
    order_counts = df["CustomerID"].value_counts()
    df["CustomerOrderCount"] = df["CustomerID"].map(order_counts)
    df["IsRepeatCustomer"] = (df["CustomerOrderCount"] > 1).astype(int)

    return df


def one_hot_encode(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """
    Translate nominal categorical columns into orthogonal coordinate
    axes (one-hot encoding), avoiding the false ordinal distance that
    label/integer encoding would introduce (e.g. Tokyo != 3 * London).
    """
    return pd.get_dummies(df, columns=columns, prefix=columns, dtype=int)


def find_multicollinear_pairs(df: pd.DataFrame, numeric_cols: list[str],
                               threshold: float = 0.80) -> pd.DataFrame:
    """
    Build the absolute correlation matrix, isolate the upper triangle,
    and return feature pairs whose correlation exceeds `threshold` --
    candidates for removal to keep the design matrix well-conditioned
    (avoids the X^T X singular-matrix problem).
    """
    corr = df[numeric_cols].corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

    pairs = []
    for col in upper.columns:
        for row in upper.index:
            val = upper.loc[row, col]
            if pd.notna(val) and val > threshold:
                pairs.append({"feature_a": row, "feature_b": col, "correlation": round(val, 3)})

    return pd.DataFrame(pairs).sort_values("correlation", ascending=False) if pairs else pd.DataFrame(
        columns=["feature_a", "feature_b", "correlation"]
    )


def resolve_multicollinearity(df: pd.DataFrame, target_col: str,
                               numeric_cols: list[str], threshold: float = 0.80) -> tuple[pd.DataFrame, list[str]]:
    """
    For every collinear pair found (correlation > threshold), keep the
    feature with the stronger correlation to the target variable and
    drop the weaker one, rather than arbitrarily dropping the first
    column encountered.
    """
    pairs = find_multicollinear_pairs(df, numeric_cols, threshold)
    dropped = []

    if pairs.empty:
        return df, dropped

    remaining = set(numeric_cols)
    for _, row in pairs.iterrows():
        a, b = row["feature_a"], row["feature_b"]
        if a not in remaining or b not in remaining:
            continue
        if target_col not in df.columns:
            # No target available to arbitrate; skip automatic drop.
            continue
        corr_a = abs(df[a].corr(df[target_col]))
        corr_b = abs(df[b].corr(df[target_col]))
        weaker = b if corr_a >= corr_b else a
        remaining.discard(weaker)
        dropped.append(weaker)

    return df.drop(columns=dropped), dropped
