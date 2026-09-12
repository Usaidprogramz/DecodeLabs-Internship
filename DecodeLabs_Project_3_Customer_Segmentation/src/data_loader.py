"""data_loader.py -- loads Project 2's order-level output as input."""
from pathlib import Path
import pandas as pd


def load_orders(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["Date"])
