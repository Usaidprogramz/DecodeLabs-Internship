"""
target_builder.py
-------------------
Constructs the binary IsFraud target used throughout Project 2.

IMPORTANT — this is a proxy label, not a genuine fraud flag.
------------------------------------------------------------
The Project 1 orders dataset has no real fraud/legitimate label. To
still practice a supervised fraud-detection pipeline on it, we treat
"Returned" or "Cancelled" orders as a stand-in for "fraudulent" --
orders that didn't complete successfully, which in a real business
often correlate with disputes, chargebacks, or bad-actor behavior,
even though most such orders are of course NOT actually fraud.

This is a legitimate technique when genuine labels aren't available,
but it must be stated plainly rather than implied: every result in
this project describes "predicting problem orders (returned or
cancelled)", not "detecting confirmed financial fraud". The README
and notebook repeat this caveat wherever results are discussed.
"""

import pandas as pd

FRAUD_STATUSES = {"Returned", "Cancelled"}


def build_fraud_target(df: pd.DataFrame, status_col: str = "OrderStatus") -> pd.DataFrame:
    """
    Add a binary `IsFraud` column: 1 if `status_col` is in FRAUD_STATUSES,
    else 0. Returns a copy; does not mutate the input dataframe.
    """
    df = df.copy()
    df["IsFraud"] = df[status_col].isin(FRAUD_STATUSES).astype(int)
    return df


def class_balance_report(df: pd.DataFrame, target_col: str = "IsFraud") -> pd.DataFrame:
    """Return a small dataframe summarizing class counts and percentages."""
    counts = df[target_col].value_counts().sort_index()
    pct = df[target_col].value_counts(normalize=True).sort_index() * 100
    report = pd.DataFrame({
        "label": ["Legitimate (0)", "Fraud-proxy (1)"],
        "count": counts.values,
        "pct": pct.round(2).values,
    })
    return report
