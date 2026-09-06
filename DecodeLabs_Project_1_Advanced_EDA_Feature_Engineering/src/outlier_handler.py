"""
outlier_handler.py
--------------------
Implements outlier detection via the Interquartile Range (IQR) method
and neutralizes them through Winsorization (capping) rather than row
deletion, preserving row count and sequential integrity as recommended
in the project brief ("Neutralizing Outliers: Winsorization vs. Deletion").

    Lower Bound = Q1 - 1.5 * IQR
    Upper Bound = Q3 + 1.5 * IQR
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class OutlierReport:
    column: str
    q1: float
    q3: float
    iqr: float
    lower_bound: float
    upper_bound: float
    n_outliers: int
    pct_outliers: float


def compute_iqr_bounds(series: pd.Series, k: float = 1.5) -> tuple[float, float, float, float]:
    """Return (Q1, Q3, lower_bound, upper_bound) for a numeric series."""
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - k * iqr
    upper = q3 + k * iqr
    return q1, q3, lower, upper


def detect_outliers(df: pd.DataFrame, columns: list[str], k: float = 1.5) -> list[OutlierReport]:
    """Detect (but do not modify) outliers in the given numeric columns."""
    reports = []
    for col in columns:
        q1, q3, lower, upper = compute_iqr_bounds(df[col], k=k)
        mask = (df[col] < lower) | (df[col] > upper)
        reports.append(OutlierReport(
            column=col,
            q1=round(q1, 2),
            q3=round(q3, 2),
            iqr=round(q3 - q1, 2),
            lower_bound=round(lower, 2),
            upper_bound=round(upper, 2),
            n_outliers=int(mask.sum()),
            pct_outliers=round(mask.mean() * 100, 2),
        ))
    return reports


def winsorize_columns(df: pd.DataFrame, columns: list[str], k: float = 1.5,
                       add_flag_columns: bool = True) -> pd.DataFrame:
    """
    Cap values outside the IQR fence at the fence boundary
    (numpy.clip), instead of deleting the row. Optionally adds a
    boolean '<col>_was_outlier' flag column so downstream models can
    still learn from "this was an extreme value" as a signal.
    """
    df = df.copy()
    for col in columns:
        q1, q3, lower, upper = compute_iqr_bounds(df[col], k=k)
        if add_flag_columns:
            df[f"{col}_was_outlier"] = ((df[col] < lower) | (df[col] > upper)).astype(int)
        df[col] = np.clip(df[col], lower, upper)
    return df


def outlier_report_to_dataframe(reports: list[OutlierReport]) -> pd.DataFrame:
    return pd.DataFrame([r.__dict__ for r in reports])
