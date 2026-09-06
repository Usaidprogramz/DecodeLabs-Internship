"""
imputation_comparison_demo.py
-------------------------------
A self-contained demonstration comparing Mean, Median, and KNN
imputation strategies, as suggested by the project brief's conclusion:

    "experiment with unique solutions -- like comparing Mean vs. KNN
    imputation to see which preserves the data distribution better."

This dataset's numeric columns (Quantity, UnitPrice, ItemsInCart,
TotalPrice) have 0% real missingness, so there is nothing genuine to
compare them on. To make the comparison meaningful, this script
artificially punches a Missing-Completely-At-Random (MCAR) hole into
UnitPrice (a controlled experiment: we KNOW the true values, so we can
directly measure how much distribution shape and error each strategy
introduces) and reports which method reconstructs the original
distribution most faithfully.

Run with:
    python -m src.imputation_comparison_demo
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.data_loader import load_raw_dataset

RAW_PATH = Path("data/raw/Dataset_for_Data_Analytics.xlsx")
REPORTS_DIR = Path("reports")
FIGURES_DIR = REPORTS_DIR / "figures"

RANDOM_SEED = 42
MISSING_FRACTION = 0.15  # 15% MCAR hole punched into UnitPrice
TARGET_COLUMN = "UnitPrice"
KNN_HELPER_COLUMNS = ["Quantity", "ItemsInCart", "TotalPrice"]


def punch_mcar_hole(series: pd.Series, fraction: float, seed: int) -> pd.Series:
    """Return a copy of `series` with `fraction` of values set to NaN,
    completely at random (MCAR) -- a controlled experiment where the
    ground truth is still known to us for scoring."""
    rng = np.random.default_rng(seed)
    corrupted = series.copy()
    n_missing = int(len(series) * fraction)
    missing_idx = rng.choice(series.index, size=n_missing, replace=False)
    corrupted.loc[missing_idx] = np.nan
    return corrupted


def run_comparison(raw_path: Path = RAW_PATH, save_outputs: bool = True) -> pd.DataFrame:
    df = load_raw_dataset(raw_path)
    true_values = df[TARGET_COLUMN].copy()

    corrupted = punch_mcar_hole(true_values, MISSING_FRACTION, RANDOM_SEED)
    missing_mask = corrupted.isna()

    results = []

    # --- Strategy 1: Mean imputation ------------------------------------
    mean_filled = corrupted.fillna(corrupted.mean())

    # --- Strategy 2: Median imputation -----------------------------------
    median_filled = corrupted.fillna(corrupted.median())

    # --- Strategy 3: KNN imputation (uses correlated numeric columns) ----
    knn_df = df[KNN_HELPER_COLUMNS + [TARGET_COLUMN]].copy()
    knn_df[TARGET_COLUMN] = corrupted
    imputer = KNNImputer(n_neighbors=5)
    knn_result = imputer.fit_transform(knn_df)
    knn_filled = pd.Series(knn_result[:, -1], index=df.index)

    strategies = {
        "Mean": mean_filled,
        "Median": median_filled,
        "KNN (k=5)": knn_filled,
    }

    for name, filled in strategies.items():
        true_missing = true_values[missing_mask]
        pred_missing = filled[missing_mask]
        results.append({
            "strategy": name,
            "MAE_on_masked_values": round(mean_absolute_error(true_missing, pred_missing), 4),
            "RMSE_on_masked_values": round(np.sqrt(mean_squared_error(true_missing, pred_missing)), 4),
            "reconstructed_std": round(filled.std(), 4),
            "original_std": round(true_values.std(), 4),
            "std_distortion_pct": round(
                abs(filled.std() - true_values.std()) / true_values.std() * 100, 2
            ),
        })

    result_df = pd.DataFrame(results).sort_values("RMSE_on_masked_values")

    if save_outputs:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        result_df.to_csv(REPORTS_DIR / "imputation_strategy_comparison.csv", index=False)
        _plot_comparison(true_values, strategies, missing_mask)

    return result_df


def _plot_comparison(true_values, strategies, missing_mask):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(strategies) + 1, figsize=(4 * (len(strategies) + 1), 4), sharey=True)

    axes[0].hist(true_values, bins=30, color="#4C72B0")
    axes[0].set_title("Original (ground truth)")

    for ax, (name, filled) in zip(axes[1:], strategies.items()):
        ax.hist(filled, bins=30, color="#DD8452")
        ax.set_title(f"{name} imputed")

    fig.suptitle(f"Distribution shape: original vs. imputed ({TARGET_COLUMN}, "
                 f"{int(missing_mask.mean() * 100)}% MCAR punched out)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "07_imputation_strategy_comparison.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    result_df = run_comparison()
    print("Imputation strategy comparison (lower RMSE / MAE / std_distortion_pct = better):\n")
    print(result_df.to_string(index=False))
    print("\nSaved: reports/imputation_strategy_comparison.csv")
    print("Saved: reports/figures/07_imputation_strategy_comparison.png")
