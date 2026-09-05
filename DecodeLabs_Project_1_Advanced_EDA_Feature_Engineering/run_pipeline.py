from pathlib import Path
from src.data_pipeline import run_pipeline, missingness_profile, correlation_audit

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw" / "Dataset_for_Data_Analytics.xlsx"
OUT = ROOT / "data" / "processed" / "cleaned_orders_features.csv"

if __name__ == "__main__":
    df = run_pipeline(RAW, OUT)
    print(f"Saved {len(df):,} rows to {OUT}")
    print("\nMissingness after processing:\n", missingness_profile(df).head())
    print("\nHigh-correlation audit (> 0.80):\n", correlation_audit(df))
