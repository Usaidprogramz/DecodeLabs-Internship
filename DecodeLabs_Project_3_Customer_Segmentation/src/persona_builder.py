"""
persona_builder.py
--------------------
Phase 4 (Translate) of the brief's IPO architecture: K-Means runs in
abstract PCA space, which means nothing to a marketing team. This
module maps cluster assignments back onto the ORIGINAL, human-readable
customer features (not the PCA coordinates) and derives a short,
actionable persona name + recommended action per cluster -- the
brief's "Strategic Persona Matrix".
"""

from dataclasses import dataclass

import pandas as pd


@dataclass
class Persona:
    cluster_id: int
    name: str
    size: int
    pct_of_customers: float
    profile: dict            # feature: mean value, for the defining features
    recommended_action: str


def attach_cluster_labels(customer_df: pd.DataFrame, labels) -> pd.DataFrame:
    df = customer_df.copy()
    df["Cluster"] = labels
    return df


def summarize_clusters(customer_df_with_clusters: pd.DataFrame, profile_columns: list[str]) -> pd.DataFrame:
    """Mean of each profiling feature, per cluster, on the ORIGINAL (unscaled) scale."""
    summary = customer_df_with_clusters.groupby("Cluster")[profile_columns].mean().round(2)
    sizes = customer_df_with_clusters["Cluster"].value_counts().sort_index()
    summary.insert(0, "ClusterSize", sizes)
    summary.insert(1, "PctOfCustomers", (sizes / sizes.sum() * 100).round(1))
    return summary


def _persona_name_and_action(row: pd.Series, overall: pd.Series) -> tuple[str, str]:
    """
    Simple, transparent rule-based naming: compare each cluster's mean
    AvgOrderValue and CouponUsageRate against the overall population
    mean to place it in one of four quadrants. This is deliberately
    readable (not a black box) so the reasoning behind each persona
    name can be checked by hand against the summary table.
    """
    high_value = row["AvgOrderValue"] >= overall["AvgOrderValue"]
    high_coupon = row["CouponUsageRate"] >= overall["CouponUsageRate"]
    high_weekend = row["WeekendOrderRate"] >= overall["WeekendOrderRate"]

    if high_value and not high_coupon:
        name = "Full-Price Big Spenders"
        action = "Prioritize for early access / premium loyalty perks — high spend without needing a discount to convert."
    elif high_value and high_coupon:
        name = "Deal-Seeking High Spenders"
        action = "Target with high-value bundle promotions — they spend a lot but are price-sensitive; well-timed offers can lift order size further."
    elif not high_value and high_coupon:
        name = "Budget-Conscious Deal Hunters"
        action = "Run frequent flash sales / coupon campaigns — this segment converts primarily on discounts."
    else:
        name = "Low-Engagement Browsers"
        action = "Re-engagement campaigns (win-back emails, small incentive) — lowest spend and low discount responsiveness suggests low overall engagement."

    if high_weekend:
        action += " Weekend-heavy ordering — time promotions for Friday/Saturday."

    return name, action


def build_personas(cluster_summary: pd.DataFrame) -> list[Persona]:
    overall = cluster_summary.drop(columns=["ClusterSize", "PctOfCustomers"]).mean()
    personas = []
    for cluster_id, row in cluster_summary.iterrows():
        name, action = _persona_name_and_action(row, overall)
        profile_cols = [c for c in cluster_summary.columns if c not in ("ClusterSize", "PctOfCustomers")]
        personas.append(Persona(
            cluster_id=int(cluster_id),
            name=name,
            size=int(row["ClusterSize"]),
            pct_of_customers=float(row["PctOfCustomers"]),
            profile={c: float(row[c]) for c in profile_cols},
            recommended_action=action,
        ))
    return personas


def personas_to_dataframe(personas: list[Persona]) -> pd.DataFrame:
    return pd.DataFrame([{
        "cluster_id": p.cluster_id,
        "persona_name": p.name,
        "size": p.size,
        "pct_of_customers": p.pct_of_customers,
        "recommended_action": p.recommended_action,
        **p.profile,
    } for p in personas])
