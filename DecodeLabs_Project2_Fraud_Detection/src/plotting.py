"""
plotting.py
------------
Renders the fraud-detection dashboard: confusion matrices, ROC curves,
precision-recall curves, Random Forest feature importance, and a
side-by-side metric comparison bar chart (accuracy deliberately
omitted, matching evaluation.py).
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from src.evaluation import EvaluationResult


def plot_confusion_matrix(ax, result: EvaluationResult):
    sns.heatmap(
        result.confusion, annot=True, fmt="d", cmap="Blues", ax=ax, cbar=False,
        xticklabels=["Legitimate", "Fraud"], yticklabels=["Legitimate", "Fraud"],
    )
    ax.set_title(f"{result.model_name} — Confusion Matrix\nF1: {result.f1:.3f}")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")


def plot_roc_curves(ax, results: list[EvaluationResult]):
    for r in results:
        ax.plot(r.fpr, r.tpr, label=f"{r.model_name} (AUC={r.roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", label="Random Classifier", alpha=0.5)
    ax.set_xlabel("False Positive Rate (1 - Specificity)")
    ax.set_ylabel("True Positive Rate (Recall)")
    ax.set_title("ROC Curves — Fraud Detection")
    ax.legend(loc="lower right", fontsize=8)


def plot_precision_recall_curves(ax, results: list[EvaluationResult]):
    for r in results:
        ax.plot(r.pr_recall, r.pr_precision, label=r.model_name)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves\n(Focus on Fraud Class)")
    ax.legend(loc="upper right", fontsize=8)


def plot_feature_importance(ax, result: EvaluationResult, top_n: int = 10):
    if result.feature_importances is None:
        ax.axis("off")
        ax.set_title(f"{result.model_name}: no feature_importances_")
        return
    top = result.feature_importances.head(top_n).sort_values()
    ax.barh(top.index, top.values, color="#4C72B0")
    ax.set_title(f"Top {top_n} Most Important Features\n({result.model_name})")
    ax.set_xlabel("Feature Importance")


def plot_metric_comparison(ax, results: list[EvaluationResult]):
    metrics = ["precision", "recall", "f1", "roc_auc"]
    labels = ["Precision", "Recall", "F1-Score", "ROC-AUC"]
    x = np.arange(len(metrics))
    width = 0.8 / len(results)

    for i, r in enumerate(results):
        values = [getattr(r, m) for m in metrics]
        ax.bar(x + i * width, values, width, label=r.model_name)

    ax.set_xticks(x + width * (len(results) - 1) / 2)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05)
    ax.set_title("Model Comparison\n(ACCURACY DELIBERATELY OMITTED)")
    ax.legend(fontsize=8)


def build_dashboard(results: list[EvaluationResult], figures_dir: Path):
    figures_dir.mkdir(parents=True, exist_ok=True)
    rf_result = next((r for r in results if r.feature_importances is not None), results[-1])

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))

    plot_confusion_matrix(axes[0, 0], results[0])
    plot_confusion_matrix(axes[0, 1], results[1] if len(results) > 1 else results[0])
    plot_roc_curves(axes[0, 2], results)
    plot_precision_recall_curves(axes[1, 0], results)
    plot_feature_importance(axes[1, 1], rf_result)
    plot_metric_comparison(axes[1, 2], results)

    fig.suptitle(
        "Fraud Detection Dashboard: confusion matrices, ROC curves, precision-recall curves, "
        "feature importance, and side-by-side model comparison.",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(figures_dir / "01_fraud_detection_dashboard.png", dpi=120)
    plt.close(fig)


def plot_class_balance(y, figures_dir: Path):
    figures_dir.mkdir(parents=True, exist_ok=True)
    counts = y.value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(["Legitimate (0)", "Fraud-proxy (1)"], counts.values, color=["#4C72B0", "#C44E52"])
    for i, v in enumerate(counts.values):
        ax.text(i, v + 5, f"{v}\n({v / counts.sum() * 100:.1f}%)", ha="center")
    ax.set_title("Class Balance — IsFraud (proxy target)")
    ax.set_ylabel("Order count")
    fig.tight_layout()
    fig.savefig(figures_dir / "00_class_balance.png", dpi=120)
    plt.close(fig)
