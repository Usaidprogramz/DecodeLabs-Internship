"""
pipeline.py
------------
Orchestrates the full customer segmentation pipeline:

  1. Load Project 2's order-level output
  2. Aggregate to one row per customer (20+ behavioral/spend features)
  3. Scale + PCA (Phase 1/2: Scale, Compress)
  4. Prove the optimal K via Elbow Method + Silhouette Score
  5. Fit final K-Means (Phase 3: Cluster)
  6. Translate clusters into business personas (Phase 4: Translate)
  7. Save outputs
"""

from pathlib import Path

import pandas as pd

from src.data_loader import load_orders
from src.customer_aggregation import build_customer_features, numeric_feature_columns
from src.dimensionality_reduction import reduce_dimensions
from src.clustering import evaluate_k_range, fit_final_clustering
from src.persona_builder import attach_cluster_labels, summarize_clusters, build_personas, personas_to_dataframe
from src.plotting import (
    plot_explained_variance, plot_elbow_and_silhouette, plot_clusters_2d, plot_persona_matrix,
)

RAW_PATH = Path("data/raw/project2_orders_with_fraud_features.csv")
PROCESSED_DIR = Path("data/processed")
REPORTS_DIR = Path("reports")
FIGURES_DIR = REPORTS_DIR / "figures"

# These columns are excluded from the CLUSTERING feature set (though
# they remain in data/processed/customer_features.csv for reference).
# Reason: 99.1% of customers placed exactly one order, so TotalOrders,
# IsRepeatCustomer, TenureDays, and StdOrderValue are constant (or
# ~zero) for 99.1% of rows and only vary for the 11 repeat customers.
# StandardScaler turns that into enormous z-scores for those 11 rows,
# which then dominates PCA and produces a trivial "repeat vs.
# single-order" split rather than a meaningful segmentation by
# spending behavior. Excluding them lets clustering find structure
# across the whole customer base instead of rediscovering the
# single-order/repeat-order split we already know about.
CLUSTERING_EXCLUDE_COLUMNS = ["TotalOrders", "IsRepeatCustomer", "TenureDays", "StdOrderValue"]

# Features shown in the persona summary table -- a readable subset of
# the full feature set, chosen to tell the business story clearly.
PROFILE_COLUMNS = [
    "AvgOrderValue", "TotalSpend", "CouponUsageRate", "WeekendOrderRate",
    "HighValueOrderRate", "FraudProxyRate", "AvgQuantity", "UniqueProducts",
]


def run_pipeline(raw_path: Path = RAW_PATH, save_outputs: bool = True) -> dict:
    # ---- 1 & 2: load + aggregate to customer level -----------------------
    df_orders = load_orders(raw_path)
    customer_df = build_customer_features(df_orders)
    feature_cols = [c for c in numeric_feature_columns(customer_df) if c not in CLUSTERING_EXCLUDE_COLUMNS]
    X = customer_df[feature_cols]

    # ---- 3: scale + PCA -----------------------------------------------------
    pca_result = reduce_dimensions(X, max_components=3)

    # ---- 4: prove optimal K -------------------------------------------------
    k_result = evaluate_k_range(pca_result.components, k_min=2, k_max=10)

    # ---- 5: fit final clustering ---------------------------------------------
    kmeans_model, labels, final_silhouette = fit_final_clustering(pca_result.components, k_result.chosen_k)

    # ---- 6: translate into personas --------------------------------------------
    customer_with_clusters = attach_cluster_labels(customer_df, labels)
    cluster_summary = summarize_clusters(customer_with_clusters, PROFILE_COLUMNS)
    personas = build_personas(cluster_summary)
    personas_df = personas_to_dataframe(personas)

    # ---- 7: save outputs --------------------------------------------------------
    if save_outputs:
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)

        customer_df.to_csv(PROCESSED_DIR / "customer_features.csv", index=False)
        customer_with_clusters.to_csv(PROCESSED_DIR / "customer_segments.csv", index=False)

        pd.DataFrame({
            "k": k_result.k_values,
            "wcss": k_result.wcss,
            "silhouette_score": k_result.silhouette_scores,
        }).to_csv(REPORTS_DIR / "k_selection_diagnostics.csv", index=False)

        pd.DataFrame([{
            "elbow_k": k_result.elbow_k,
            "best_silhouette_k": k_result.best_silhouette_k,
            "chosen_k": k_result.chosen_k,
            "final_silhouette_score": round(final_silhouette, 4),
            "n_pca_components_kept": pca_result.n_components_kept,
            "n_components_for_95pct_variance": pca_result.n_components_95pct,
            "cumulative_variance_kept": round(pca_result.cumulative_variance[-1], 4),
        }]).to_csv(REPORTS_DIR / "clustering_summary.csv", index=False)

        cluster_summary.to_csv(REPORTS_DIR / "cluster_profile_summary.csv")
        personas_df.to_csv(REPORTS_DIR / "customer_personas.csv", index=False)

        plot_explained_variance(pca_result.cumulative_variance, pca_result.n_components_95pct, FIGURES_DIR)
        plot_elbow_and_silhouette(k_result, FIGURES_DIR)
        plot_clusters_2d(pca_result.components, labels, FIGURES_DIR)
        plot_persona_matrix(personas_df, FIGURES_DIR)

    return {
        "df_orders": df_orders,
        "customer_df": customer_df,
        "feature_cols": feature_cols,
        "pca_result": pca_result,
        "k_result": k_result,
        "kmeans_model": kmeans_model,
        "labels": labels,
        "final_silhouette": final_silhouette,
        "customer_with_clusters": customer_with_clusters,
        "cluster_summary": cluster_summary,
        "personas": personas,
        "personas_df": personas_df,
    }


if __name__ == "__main__":
    outcome = run_pipeline()
    print(f"Customers: {len(outcome['customer_df'])}")
    print(f"Features engineered (pre-PCA): {len(outcome['feature_cols'])}")
    print(f"PCA components kept: {outcome['pca_result'].n_components_kept} "
          f"(cumulative variance: {outcome['pca_result'].cumulative_variance[-1]:.3f})")
    print(f"Chosen K: {outcome['k_result'].chosen_k} "
          f"(elbow suggested {outcome['k_result'].elbow_k}, "
          f"silhouette suggested {outcome['k_result'].best_silhouette_k})")
    print(f"Final silhouette score: {outcome['final_silhouette']:.4f}")
    print()
    print(outcome["personas_df"][["cluster_id", "persona_name", "size", "pct_of_customers"]].to_string(index=False))
