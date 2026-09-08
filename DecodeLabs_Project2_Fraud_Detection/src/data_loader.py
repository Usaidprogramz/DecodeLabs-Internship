"""
data_loader.py
----------------
Loads the Project 1 cleaned output as the input to Project 2. Project 2
does not re-clean raw data -- Project 1 already handled missingness,
outliers, and base feature engineering. This module's only job is to
load that trusted, already-validated dataset.
"""

from pathlib import Path

import pandas as pd


def load_project1_dataset(path: Path) -> pd.DataFrame:
    """Load the cleaned Project 1 dataset (data/raw/project1_cleaned_dataset.csv)."""
    df = pd.read_csv(path, parse_dates=["Date"])
    return df
