"""
pipeline.py
------------
Orchestrates the full leak-free fraud detection pipeline:

  1. Load Project 1's cleaned dataset
  2. Build the IsFraud proxy target
  3. Engineer fraud-specific features
  4. Split 80/20, STRATIFIED, BEFORE any resampling or scaling
  5. Tune Logistic Regression + Random Forest (SMOTE inside each,
     leak-free via imblearn.pipeline.Pipeline + GridSearchCV)
  6. Evaluate both on the untouched test set (Precision/Recall/F1/ROC-AUC)
  7. Save the dashboard, metrics table, and processed dataset
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.data_loader import load_project1_dataset
from src.target_builder import build_fraud_target, class_balance_report
from src.feature_engineering import engineer_fraud_features, build_feature_matrix
from src.modeling import (
    build_logistic_regression_pipeline,
    build_random_forest_pipeline,
    LOGISTIC_REGRESSION_PARAM_GRID,
    RANDOM_FOREST_PARAM_GRID,
    tune_model,
    RANDOM_STATE,
)
from src.evaluation import evaluate_model, results_to_dataframe
from src.plotting import build_dashboard, plot_class_balance

RAW_PATH = Path("data/raw/project1_cleaned_dataset.csv")
PROCESSED_DIR = Path("data/processed")
REPORTS_DIR = Path("reports")
FIGURES_DIR = REPORTS_DIR / "figures"

TEST_SIZE = 0.2


def run_pipeline(raw_path: Path = RAW_PATH, save_outputs: bool = True) -> dict:
    # ---- 1 & 2: load + target ------------------------------------------
    df_raw = load_project1_dataset(raw_path)
    df_labeled = build_fraud_target(df_raw)
    balance = class_balance_report(df_labeled)

    # ---- 3: feature engineering -----------------------------------------
    df_features = engineer_fraud_features(df_labeled)
    X, y = build_feature_matrix(df_features)

    # ---- 4: stratified split BEFORE any resampling/scaling --------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    # ---- 5: tune both models (SMOTE stays inside each CV fold) ----------
    lr_tuned = tune_model(
        build_logistic_regression_pipeline(),
        LOGISTIC_REGRESSION_PARAM_GRID,
        X_train, y_train,
        name="Logistic Regression",
    )
    rf_tuned = tune_model(
        build_random_forest_pipeline(),
        RANDOM_FOREST_PARAM_GRID,
        X_train, y_train,
        name="Random Forest",
    )

    # ---- 6: evaluate on the untouched test set ---------------------------
    lr_result = evaluate_model(lr_tuned.best_estimator, X_test, y_test, "Logistic Regression")
    rf_result = evaluate_model(rf_tuned.best_estimator, X_test, y_test, "Random Forest")
    results = [lr_result, rf_result]
    metrics_df = results_to_dataframe(results)

    best_model_name = metrics_df.sort_values("recall", ascending=False).iloc[0]["model"]

    # ---- 7: save outputs ---------------------------------------------------
    if save_outputs:
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)

        df_features.to_csv(PROCESSED_DIR / "orders_with_fraud_features.csv", index=False)
        balance.to_csv(REPORTS_DIR / "class_balance.csv", index=False)
        metrics_df.to_csv(REPORTS_DIR / "model_evaluation_metrics.csv", index=False)

        pd.DataFrame([
            {"model": "Logistic Regression", **lr_tuned.best_params, "best_cv_recall": round(lr_tuned.best_cv_recall, 4)},
        ]).to_csv(REPORTS_DIR / "logistic_regression_best_params.csv", index=False)
        pd.DataFrame([
            {"model": "Random Forest", **rf_tuned.best_params, "best_cv_recall": round(rf_tuned.best_cv_recall, 4)},
        ]).to_csv(REPORTS_DIR / "random_forest_best_params.csv", index=False)

        if rf_result.feature_importances is not None:
            rf_result.feature_importances.rename("importance").rename_axis("feature").to_csv(
                REPORTS_DIR / "random_forest_feature_importance.csv"
            )

        plot_class_balance(y, FIGURES_DIR)
        build_dashboard(results, FIGURES_DIR)

    return {
        "df_features": df_features,
        "X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test,
        "balance": balance,
        "lr_tuned": lr_tuned, "rf_tuned": rf_tuned,
        "results": results,
        "metrics_df": metrics_df,
        "best_model_name": best_model_name,
    }


if __name__ == "__main__":
    outcome = run_pipeline()
    print(f"Best model by recall: {outcome['best_model_name']}")
    print(outcome["metrics_df"].to_string(index=False))
