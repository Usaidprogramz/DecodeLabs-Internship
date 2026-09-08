"""
main.py
--------
Single entry point for Project 2: Fraud Detection Pipeline.

Run with:
    python main.py
"""

from src.pipeline import run_pipeline


def main():
    print("=" * 70)
    print("DecodeLabs | Project 2: Fraud Detection Pipeline")
    print("=" * 70)
    print()
    print("NOTE: 'IsFraud' is a PROXY target (Returned/Cancelled orders),")
    print("not a genuine fraud label. See README.md for why.")
    print()

    outcome = run_pipeline()

    balance = outcome["balance"]
    print("Class balance (proxy IsFraud target):")
    print(balance.to_string(index=False))
    print()

    print(f"Train rows: {len(outcome['X_train'])}   Test rows: {len(outcome['X_test'])}")
    print(f"Features used: {outcome['X_train'].shape[1]}")
    print()

    print("Best hyperparameters found (GridSearchCV, optimizing recall):")
    print(f"  Logistic Regression: {outcome['lr_tuned'].best_params}")
    print(f"    -> best CV recall: {outcome['lr_tuned'].best_cv_recall:.3f}")
    print(f"  Random Forest:       {outcome['rf_tuned'].best_params}")
    print(f"    -> best CV recall: {outcome['rf_tuned'].best_cv_recall:.3f}")
    print()

    print("Test-set evaluation (accuracy deliberately excluded):")
    print(outcome["metrics_df"].to_string(index=False))
    print()

    print(f"Best model by recall: {outcome['best_model_name']}")
    print()

    print("Outputs written to:")
    print("  data/processed/orders_with_fraud_features.csv")
    print("  reports/class_balance.csv")
    print("  reports/model_evaluation_metrics.csv")
    print("  reports/logistic_regression_best_params.csv")
    print("  reports/random_forest_best_params.csv")
    print("  reports/random_forest_feature_importance.csv")
    print("  reports/figures/00_class_balance.png")
    print("  reports/figures/01_fraud_detection_dashboard.png")


if __name__ == "__main__":
    main()
