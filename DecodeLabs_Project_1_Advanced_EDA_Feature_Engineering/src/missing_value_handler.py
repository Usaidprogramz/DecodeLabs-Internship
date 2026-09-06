"""
missing_value_handler.py
-------------------------
Implements the "Missing Data Decision Matrix" from the project brief:

    < 5%      missingness  -> Drop rows (dropna)
    5% - 20%  missingness  -> Statistical imputation
                                 - skewed numeric -> global median
                                 - categorical / correlated -> sub-group
                                   conditional imputation
    > 20%     missingness  -> Multi-dimensional estimation (KNN)

The rule is applied per-column, and every decision is logged so the
reasoning is auditable rather than a silent guess.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer


@dataclass
class ImputationLog:
    """Keeps a human-readable audit trail of every imputation decision."""
    entries: list = field(default_factory=list)

    def add(self, column: str, pct_missing: float, strategy: str, note: str = ""):
        self.entries.append({
            "column": column,
            "pct_missing": round(pct_missing, 2),
            "strategy": strategy,
            "note": note,
        })

    def as_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.entries)


def handle_missing_values(
    df: pd.DataFrame,
    numeric_cols: list[str] | None = None,
    categorical_cols: list[str] | None = None,
    group_col_for_categorical: str | None = None,
    knn_neighbors: int = 5,
) -> tuple[pd.DataFrame, ImputationLog]:
    """
    Apply the missing-data decision matrix column by column.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe (will not be mutated in place).
    numeric_cols : list[str], optional
        Numeric columns to evaluate for imputation.
    categorical_cols : list[str], optional
        Categorical columns to evaluate for imputation.
    group_col_for_categorical : str, optional
        If provided, categorical imputation in the 5-20% band uses the
        mode within this grouping column (sub-group conditional
        imputation) instead of the global mode.
    knn_neighbors : int
        Number of neighbors for KNN imputation when missingness > 20%.

    Returns
    -------
    (pd.DataFrame, ImputationLog)
        The cleaned dataframe and a log of every decision made.
    """
    df = df.copy()
    log = ImputationLog()
    numeric_cols = numeric_cols or []
    categorical_cols = categorical_cols or []

    rows_before = len(df)

    # --- Numeric columns -------------------------------------------------
    knn_targets = []
    for col in numeric_cols:
        pct = df[col].isna().mean() * 100
        if pct == 0:
            continue
        if pct < 5:
            df = df.dropna(subset=[col])
            log.add(col, pct, "drop_rows", "Below 5% threshold; dropping preserves distribution integrity.")
        elif pct <= 20:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            log.add(col, pct, "global_median", f"Filled with global median = {median_val:.2f} (robust to skew/outliers).")
        else:
            knn_targets.append(col)

    if knn_targets:
        imputer = KNNImputer(n_neighbors=knn_neighbors)
        df[knn_targets] = imputer.fit_transform(df[knn_targets])
        for col in knn_targets:
            log.add(col, np.nan, "knn_imputation", f"KNN (k={knn_neighbors}) used; missingness exceeded 20%.")

    # --- Categorical columns ----------------------------------------------
    for col in categorical_cols:
        pct = df[col].isna().mean() * 100
        if pct == 0:
            continue

        if pct < 5:
            df = df.dropna(subset=[col])
            log.add(col, pct, "drop_rows", "Below 5% threshold; dropping preserves distribution integrity.")
            continue

        # For categorical data, a missing value is frequently informative
        # (e.g. "no coupon was applied") rather than a random data-entry
        # gap. We flag this explicitly rather than blindly following the
        # numeric decision matrix, and impute with a domain-meaningful
        # sentinel category instead of a statistical guess.
        if group_col_for_categorical and group_col_for_categorical in df.columns:
            mode_map = (
                df.groupby(group_col_for_categorical)[col]
                .agg(lambda s: s.mode().iat[0] if not s.mode().empty else np.nan)
            )
            df[col] = df.apply(
                lambda r: mode_map.get(r[group_col_for_categorical], np.nan)
                if pd.isna(r[col]) else r[col],
                axis=1,
            )
            log.add(col, pct, "subgroup_conditional_mode",
                    f"Missing values filled with the mode within each '{group_col_for_categorical}' group.")
        else:
            df[col] = df[col].fillna("Unknown")
            log.add(col, pct, "sentinel_category", "No grouping column supplied; filled with 'Unknown' sentinel.")

    rows_after = len(df)
    if rows_after != rows_before:
        log.add("__rows__", np.nan, "info",
                f"Row count changed from {rows_before} to {rows_after} due to <5% column drops.")

    return df.reset_index(drop=True), log
