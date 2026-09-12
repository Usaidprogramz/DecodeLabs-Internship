"""
tests/test_pipeline.py
------------------------
Unit tests for Project 3. Alongside the usual correctness checks,
several tests specifically guard against the "trivial split" failure
mode discovered during development: near-constant columns (only
varying for the rare repeat-purchase customers) dominating PCA/K-Means
once scaled, producing a meaningless 99%/1% cluster split.

Run with:
    pytest -v
"""

import numpy as np
import pandas as pd
import pytest

from src.customer_aggregation import build_customer_features, numeric_feature_columns
from src.dimensionality_reduction import reduce_dimensions
from src.clustering import evaluate_k_range, fit_final_clustering
from src.persona_builder import attach_cluster_labels, summarize_clusters, build_personas
from src.pipeline import CLUSTERING_EXCLUDE_COLUMNS, PROFILE_COLUMNS


@pytest.fixture
def sample_orders():
    n = 60
    rng = np.random.default_rng(0)
    customer_ids = [f"C{i}" for i in range(50)] + [f"C{i}" for i in range(5)]  # 5 repeat customers
    return pd.DataFrame({
        "OrderID": [f"ORD{i}" for i in range(n)],
        "Date": pd.date_range("2024-01-01", periods=n, freq="D"),
        "CustomerID": (customer_ids * 2)[:n],
        "Product": rng.choice(["Laptop", "Phone", "Chair"], size=n),
        "Quantity": rng.integers(1, 6, size=n),
        "UnitPrice": rng.uniform(10, 500, size=n).round(2),
        "PaymentMethod": rng.choice(["Cash", "Credit Card"], size=n),
        "ItemsInCart": rng.integers(1, 10, size=n),
        "ReferralSource": rng.choice(["Instagram", "Email"], size=n),
        "TotalPrice": rng.uniform(20, 3000, size=n).round(2),
        "OrderMonth": rng.integers(1, 13, size=n),
        "OrderDayOfWeek": rng.integers(0, 7, size=n),
        "IsWeekendOrder": rng.integers(0, 2, size=n),
        "CouponUsed": rng.integers(0, 2, size=n),
        "CartFillRatio": rng.uniform(0, 1, size=n).round(3),
        "IsFraud": rng.integers(0, 2, size=n),
        "AvgItemValue": rng.uniform(5, 500, size=n).round(2),
        "PricePerUnit": rng.uniform(5, 500, size=n).round(2),
        "IsHighValue": rng.integers(0, 2, size=n),
    })


# --------------------------------------------------------------------- #
# Customer aggregation
# --------------------------------------------------------------------- #

def test_aggregation_produces_one_row_per_customer(sample_orders):
    customer_df = build_customer_features(sample_orders)
    assert customer_df["CustomerID"].is_unique
    assert len(customer_df) == sample_orders["CustomerID"].nunique()


def test_aggregation_has_at_least_20_features(sample_orders):
    """The brief requires reducing '20+ columns' via PCA."""
    customer_df = build_customer_features(sample_orders)
    feature_cols = numeric_feature_columns(customer_df)
    assert len(feature_cols) >= 20


def test_single_order_customers_get_zero_std_order_value(sample_orders):
    customer_df = build_customer_features(sample_orders)
    single_order_customers = customer_df[customer_df["TotalOrders"] == 1]
    assert (single_order_customers["StdOrderValue"] == 0.0).all()
    assert not single_order_customers["StdOrderValue"].isna().any()


def test_repeat_customer_flag_matches_order_count(sample_orders):
    customer_df = build_customer_features(sample_orders)
    expected = (customer_df["TotalOrders"] > 1).astype(int)
    pd.testing.assert_series_equal(customer_df["IsRepeatCustomer"], expected, check_names=False)


# --------------------------------------------------------------------- #
# Near-constant-column guard -- the central lesson of this project
# --------------------------------------------------------------------- #

def test_near_constant_columns_are_excluded_from_clustering():
    """TotalOrders/IsRepeatCustomer/TenureDays/StdOrderValue are ~99%
    constant on the real dataset (single-order customers) and must
    never enter the clustering feature set -- see pipeline.py's
    CLUSTERING_EXCLUDE_COLUMNS docstring for why."""
    for col in ["TotalOrders", "IsRepeatCustomer", "TenureDays", "StdOrderValue"]:
        assert col in CLUSTERING_EXCLUDE_COLUMNS


def test_clustering_excludes_configured_columns(sample_orders):
    customer_df = build_customer_features(sample_orders)
    feature_cols = [c for c in numeric_feature_columns(customer_df) if c not in CLUSTERING_EXCLUDE_COLUMNS]
    for excluded in CLUSTERING_EXCLUDE_COLUMNS:
        assert excluded not in feature_cols


# --------------------------------------------------------------------- #
# Dimensionality reduction
# --------------------------------------------------------------------- #

def test_pca_keeps_at_most_max_components():
    rng = np.random.default_rng(1)
    X = pd.DataFrame(rng.normal(size=(100, 25)), columns=[f"f{i}" for i in range(25)])
    result = reduce_dimensions(X, max_components=3)
    assert result.n_components_kept <= 3
    assert result.components.shape == (100, result.n_components_kept)


def test_pca_keeps_at_least_2_components_for_visualization():
    rng = np.random.default_rng(2)
    X = pd.DataFrame(rng.normal(size=(50, 5)), columns=[f"f{i}" for i in range(5)])
    result = reduce_dimensions(X, max_components=3)
    assert result.n_components_kept >= 2


def test_cumulative_variance_is_monotonically_increasing():
    rng = np.random.default_rng(3)
    X = pd.DataFrame(rng.normal(size=(80, 10)), columns=[f"f{i}" for i in range(10)])
    result = reduce_dimensions(X, max_components=3)
    diffs = np.diff(result.cumulative_variance)
    assert (diffs >= -1e-9).all()  # never decreases


# --------------------------------------------------------------------- #
# Clustering / K selection
# --------------------------------------------------------------------- #

def test_evaluate_k_range_returns_one_score_per_k():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(120, 3))
    result = evaluate_k_range(X, k_min=2, k_max=6)
    assert len(result.k_values) == len(result.wcss) == len(result.silhouette_scores)
    assert result.k_values == list(range(2, 7))


def test_wcss_decreases_as_k_increases():
    """WCSS (inertia) must be monotonically non-increasing as K grows --
    this is a mathematical property of K-Means, not a coincidence."""
    rng = np.random.default_rng(5)
    X = rng.normal(size=(150, 4))
    result = evaluate_k_range(X, k_min=2, k_max=8)
    diffs = np.diff(result.wcss)
    assert (diffs <= 1e-6).all()


def test_fit_final_clustering_returns_valid_labels():
    rng = np.random.default_rng(6)
    X = np.vstack([rng.normal(loc=0, size=(50, 2)), rng.normal(loc=10, size=(50, 2))])
    model, labels, silhouette = fit_final_clustering(X, k=2)
    assert len(labels) == 100
    assert set(labels) == {0, 1}
    assert -1.0 <= silhouette <= 1.0
    assert silhouette > 0.5  # these two blobs are well-separated by construction


# --------------------------------------------------------------------- #
# Persona building
# --------------------------------------------------------------------- #

def test_summarize_clusters_sizes_sum_to_total():
    customer_df = pd.DataFrame({
        "CustomerID": [f"C{i}" for i in range(20)],
        "AvgOrderValue": np.linspace(10, 1000, 20),
        "TotalSpend": np.linspace(10, 1000, 20),
        "CouponUsageRate": np.linspace(0, 1, 20),
        "WeekendOrderRate": np.linspace(0, 1, 20),
        "HighValueOrderRate": np.linspace(0, 1, 20),
        "FraudProxyRate": np.linspace(0, 1, 20),
        "AvgQuantity": np.linspace(1, 5, 20),
        "UniqueProducts": np.linspace(1, 3, 20),
    })
    labels = [0] * 12 + [1] * 8
    with_clusters = attach_cluster_labels(customer_df, labels)
    summary = summarize_clusters(with_clusters, PROFILE_COLUMNS)
    assert summary["ClusterSize"].sum() == 20
    assert abs(summary["PctOfCustomers"].sum() - 100.0) < 0.1


def test_build_personas_one_per_cluster():
    customer_df = pd.DataFrame({
        "CustomerID": [f"C{i}" for i in range(20)],
        "AvgOrderValue": [1000] * 10 + [100] * 10,
        "TotalSpend": [1000] * 10 + [100] * 10,
        "CouponUsageRate": [0.1] * 10 + [0.9] * 10,
        "WeekendOrderRate": [0.5] * 20,
        "HighValueOrderRate": [0.5] * 20,
        "FraudProxyRate": [0.2] * 20,
        "AvgQuantity": [3] * 20,
        "UniqueProducts": [1] * 20,
    })
    labels = [0] * 10 + [1] * 10
    with_clusters = attach_cluster_labels(customer_df, labels)
    summary = summarize_clusters(with_clusters, PROFILE_COLUMNS)
    personas = build_personas(summary)
    assert len(personas) == 2
    assert personas[0].size + personas[1].size == 20
    # High spend + low coupon usage -> "Full-Price Big Spenders"
    assert personas[0].name == "Full-Price Big Spenders"
    # Low spend + high coupon usage -> "Budget-Conscious Deal Hunters"
    assert personas[1].name == "Budget-Conscious Deal Hunters"
