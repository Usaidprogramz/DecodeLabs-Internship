"""
main.py
--------
Single entry point for Project 3: Customer Segmentation.

Run with:
    python main.py
"""

from src.pipeline import run_pipeline


def main():
    print("=" * 70)
    print("DecodeLabs | Project 3: Unsupervised Learning (Customer Segmentation)")
    print("=" * 70)
    print()
    print("NOTE: this dataset has ~1 order per customer on average, so")
    print("clusters mainly reflect single-transaction behavior, not")
    print("repeat-purchase loyalty history. See README.md.")
    print()

    outcome = run_pipeline()

    print(f"Customers: {len(outcome['customer_df'])}")
    print(f"Features engineered (pre-PCA): {len(outcome['feature_cols'])}")
    print()

    pca = outcome["pca_result"]
    print(f"PCA components needed for 95% variance: {pca.n_components_95pct}")
    print(f"PCA components kept for clustering (capped at 3): {pca.n_components_kept}")
    print(f"Cumulative variance explained by kept components: {pca.cumulative_variance[-1]:.3f}")
    print()

    k_result = outcome["k_result"]
    print("K selection (Elbow Method + Silhouette Score):")
    print(f"  Elbow Method suggests K = {k_result.elbow_k}")
    print(f"  Silhouette Score suggests K = {k_result.best_silhouette_k}")
    print(f"  Chosen K = {k_result.chosen_k}")
    print(f"  Final silhouette score at chosen K: {outcome['final_silhouette']:.4f}")
    print()

    print("Cluster profile summary (original feature scale):")
    print(outcome["cluster_summary"].to_string())
    print()

    print("Business Personas:")
    for p in outcome["personas"]:
        print(f"  Cluster {p.cluster_id}: {p.name}  "
              f"({p.size} customers, {p.pct_of_customers}%)")
        print(f"    -> {p.recommended_action}")
    print()

    print("Outputs written to:")
    print("  data/processed/customer_features.csv")
    print("  data/processed/customer_segments.csv")
    print("  reports/k_selection_diagnostics.csv")
    print("  reports/clustering_summary.csv")
    print("  reports/cluster_profile_summary.csv")
    print("  reports/customer_personas.csv")
    print("  reports/figures/01_pca_explained_variance.png")
    print("  reports/figures/02_elbow_and_silhouette.png")
    print("  reports/figures/03_clusters_pca_2d.png")
    print("  reports/figures/04_persona_matrix.png")


if __name__ == "__main__":
    main()
