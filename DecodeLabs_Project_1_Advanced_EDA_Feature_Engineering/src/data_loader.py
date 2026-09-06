"""
data_loader.py
---------------
Handles ingestion of the raw dataset and basic structural inspection.
This is the entry point of PHASE 1 (Securing Input Fidelity) in the
Input -> Process -> Output pipeline.
"""

from pathlib import Path
import pandas as pd


def load_raw_dataset(path: str | Path) -> pd.DataFrame:
    """
    Load the raw e-commerce orders dataset from an Excel file.

    Parameters
    ----------
    path : str or Path
        Path to the raw .xlsx file.

    Returns
    -------
    pd.DataFrame
        The dataset exactly as it exists on disk, with dates parsed.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Raw dataset not found at: {path}")

    df = pd.read_excel(path)

    # Ensure Date column is a proper datetime dtype (defensive parsing;
    # some spreadsheet exports store dates as text).
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    return df


def dataset_overview(df: pd.DataFrame) -> pd.DataFrame:
    """
    Produce a single-glance structural summary of a dataframe:
    dtype, missing count, missing %, and unique value count per column.

    Used at the start of EDA to decide which columns need which
    treatment (per the Missing Data Decision Matrix).
    """
    summary = pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "n_missing": df.isna().sum(),
        "pct_missing": (df.isna().mean() * 100).round(2),
        "n_unique": df.nunique(),
    })
    summary.index.name = "column"
    return summary.sort_values("pct_missing", ascending=False)
