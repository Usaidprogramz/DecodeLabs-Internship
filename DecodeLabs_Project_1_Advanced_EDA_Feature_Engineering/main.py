"""
main.py
--------
Single entry point for the DecodeLabs Data Science Project 1 pipeline.

Usage:
    python main.py
"""

from src.pipeline import run_pipeline


def main():
    print("=" * 70)
    print("DecodeLabs | Project 1: Advanced EDA & Feature Engineering")
    print("=" * 70)

    results = run_pipeline()

    print(f"\nRaw rows            : {len(results['raw'])}")
    print(f"Cleaned rows         : {len(results['cleaned'])}")
    print(f"Raw columns          : {len(results['raw'].columns)}")
    print(f"Cleaned columns      : {len(results['cleaned'].columns)}  "
          f"(+{len(results['cleaned'].columns) - len(results['raw'].columns)} engineered)")
    print(f"Multicollinear pairs found (|r| > 0.80): {len(results['collinear_pairs'])}")
    print(f"Schema validation passed: {results['validation_result'].passed}")
    print(f"Numeric features standardized (StandardScaler): {len(results['scaling_log'].entries)}")

    print("\nImputation decisions:")
    print(results["imputation_log"].as_dataframe().to_string(index=False))

    print("\nOutputs written to:")
    print("  data/processed/cleaned_dataset.csv")
    print("  data/processed/ml_ready_encoded_dataset.csv")
    print("  data/processed/ml_ready_scaled_dataset.csv")
    print("  reports/*.csv")


if __name__ == "__main__":
    main()
