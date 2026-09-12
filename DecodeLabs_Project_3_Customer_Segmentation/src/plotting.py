"""
plotting.py
------------
Renders the segmentation diagnostics: cumulative explained variance
(PCA), the Elbow Method curve, the Silhouette Score curve, and a
2D/3D scatter of clusters in PCA space.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_explained_variance(cumulative_variance, n_components_95pct, figures_dir: Path):
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(1, len(cumulative_variance) + 1)
    ax.plot(x, cumulative_variance, marker="o", color="#4C72B0")
    ax.axhline(0.95, color="#C44E52", linestyle="--", label="95% threshold")
    ax.axvline(n_components_95pct, color="#C44E52", linestyle=":", alpha=0.6)
    ax.set_xlabel("Number of Principal Components")
    ax.set_ylabel("Cumulative Explained Variance")
    ax.set_title("PCA: Cumulative Explained Variance")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "01_pca_explained_variance.png", dpi=120)
    plt.close(fig)


def plot_elbow_and_silhouette(k_result, figures_dir: Path):
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(k_result.k_values, k_result.wcss, marker="o", color="#4C72B0")
    axes[0].axvline(k_result.elbow_k, color="#C44E52", linestyle="--", label=f"Elbow k={k_result.elbow_k}")
    axes[0].set_xlabel("Number of Clusters (K)")
    axes[0].set_ylabel("WCSS (Inertia)")
    axes[0].set_title("Elbow Method")
    axes[0].legend()

    axes[1].plot(k_result.k_values, k_result.silhouette_scores, marker="o", color="#55A868")
    axes[1].axvline(k_result.best_silhouette_k, color="#C44E52", linestyle="--",
                     label=f"Best k={k_result.best_silhouette_k}")
    axes[1].set_xlabel("Number of Clusters (K)")
    axes[1].set_ylabel("Silhouette Score")
    axes[1].set_title("Silhouette Score")
    axes[1].legend()

    fig.suptitle("Diagnostic Gatekeepers: Proving the Optimal K")
    fig.tight_layout()
    fig.savefig(figures_dir / "02_elbow_and_silhouette.png", dpi=120)
    plt.close(fig)


def plot_clusters_2d(components, labels, figures_dir: Path):
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 6))
    scatter = ax.scatter(components[:, 0], components[:, 1], c=labels, cmap="tab10", alpha=0.7, s=25)
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.set_title("Customer Segments in PCA Space")
    legend = ax.legend(*scatter.legend_elements(), title="Cluster")
    ax.add_artist(legend)
    fig.tight_layout()
    fig.savefig(figures_dir / "03_clusters_pca_2d.png", dpi=120)
    plt.close(fig)


def plot_persona_matrix(personas_df, figures_dir: Path):
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    axes[0].bar(personas_df["persona_name"], personas_df["size"], color="#4C72B0")
    axes[0].set_ylabel("Number of customers")
    axes[0].set_title("Persona Sizes")
    axes[0].tick_params(axis="x", rotation=25)

    axes[1].bar(personas_df["persona_name"], personas_df["AvgOrderValue"], color="#DD8452")
    axes[1].set_ylabel("Average Order Value ($)")
    axes[1].set_title("Average Order Value by Persona")
    axes[1].tick_params(axis="x", rotation=25)

    fig.tight_layout()
    fig.savefig(figures_dir / "04_persona_matrix.png", dpi=120)
    plt.close(fig)
