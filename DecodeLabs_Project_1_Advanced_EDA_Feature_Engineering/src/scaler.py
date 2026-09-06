"""
scaler.py
----------
Final PHASE 2 step: standardizes numeric features to zero mean / unit
variance with scikit-learn's StandardScaler.

Why this runs LAST, after outlier winsorization and feature
engineering: StandardScaler fits on mean and standard deviation, both
of which are themselves sensitive to outliers. Scaling BEFORE
neutralizing outliers would let a handful of extreme values distort
the scale for every other row. Scaling AFTER winsorization means the
mean/std the scaler learns already reflect the capped, outlier-safe
distribution.

Only continuous numeric features are scaled. Binary/flag columns
(0/1 indicators such as IsWeekendOrder, CouponUsed, *_was_outlier)
and one-hot encoded columns are left untouched -- scaling a binary
flag doesn't help most estimators and makes the column harder to
read as a plain indicator.
"""

from dataclasses import dataclass, field

import pandas as pd
from sklearn.preprocessing import StandardScaler


@dataclass
class ScalingLog:
    """Audit trail of which columns were scaled and their pre-scale stats."""
    entries: list = field(default_factory=list)

    def add(self, column: str, mean: float, std: float):
        self.entries.append({
            "column": column,
            "pre_scale_mean": round(mean, 4),
            "pre_scale_std": round(std, 4),
        })

    def as_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.entries)


def scale_numeric_features(
    df: pd.DataFrame,
    columns: list[str],
) -> tuple[pd.DataFrame, StandardScaler, ScalingLog]:
    """
    Standardize the given numeric columns to zero mean / unit variance.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe (not mutated in place).
    columns : list[str]
        Continuous numeric columns to scale. Binary flags and
        one-hot encoded columns should NOT be passed here.

    Returns
    -------
    (pd.DataFrame, StandardScaler, ScalingLog)
        The dataframe with `columns` replaced by their scaled values,
        the fitted scaler (so new data can be transformed consistently
        at inference time), and a log of what was scaled.
    """
    df = df.copy()
    log = ScalingLog()

    scaler = StandardScaler()
    pre_means = df[columns].mean()
    pre_stds = df[columns].std()

    df[columns] = scaler.fit_transform(df[columns])

    for col in columns:
        log.add(col, pre_means[col], pre_stds[col])

    return df, scaler, log
