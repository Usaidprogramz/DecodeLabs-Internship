"""
customer_aggregation.py
--------------------------
Segmentation must operate on ONE ROW PER CUSTOMER, not one row per
order. This module aggregates the order-level dataset (from Projects
1 & 2) into a customer-level feature table.

Honesty note (read before changing thresholds/features here):
------------------------------------------------------------------
This dataset has ~1,200 orders across ~1,189 unique customers -- i.e.
almost every customer placed exactly one order. That means
frequency/recency-style RFM features carry very little signal here
(most customers get a "1 order" and "0 variability" value by
construction). Rather than pretend otherwise, this module still
builds those columns (so the pipeline is general and would work
properly on a dataset with real repeat-purchase history), but the
resulting personas are best understood as "single-transaction
profiles" -- what a customer bought, how much they spent, and when --
not loyalty segments. This caveat is repeated in the README and
notebook wherever cluster results are interpreted.
"""

import numpy as np
import pandas as pd

TOP_N_CATEGORICAL = 5  # one-hot only the top-N most common values per categorical column


def build_customer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate order-level rows into one row per CustomerID with 20+
    numeric behavioral/spend features plus one-hot encoded top
    categorical preferences.
    """
    reference_date = df["Date"].max() + pd.Timedelta(days=1)

    grouped = df.groupby("CustomerID")

    agg = grouped.agg(
        TotalOrders=("OrderID", "count"),
        TotalSpend=("TotalPrice", "sum"),
        AvgOrderValue=("TotalPrice", "mean"),
        StdOrderValue=("TotalPrice", "std"),
        AvgQuantity=("Quantity", "mean"),
        AvgUnitPrice=("UnitPrice", "mean"),
        AvgItemsInCart=("ItemsInCart", "mean"),
        AvgCartFillRatio=("CartFillRatio", "mean"),
        AvgPricePerUnit=("PricePerUnit", "mean"),
        AvgItemValue=("AvgItemValue", "mean"),
        CouponUsageRate=("CouponUsed", "mean"),
        WeekendOrderRate=("IsWeekendOrder", "mean"),
        HighValueOrderRate=("IsHighValue", "mean"),
        FraudProxyRate=("IsFraud", "mean"),
        UniqueProducts=("Product", "nunique"),
        UniquePaymentMethods=("PaymentMethod", "nunique"),
        FirstOrderDate=("Date", "min"),
        LastOrderDate=("Date", "max"),
        AvgOrderMonth=("OrderMonth", "mean"),
        AvgOrderDayOfWeek=("OrderDayOfWeek", "mean"),
    )

    agg["StdOrderValue"] = agg["StdOrderValue"].fillna(0.0)  # single-order customers: no variability
    agg["RecencyDays"] = (reference_date - agg["LastOrderDate"]).dt.days
    agg["TenureDays"] = (agg["LastOrderDate"] - agg["FirstOrderDate"]).dt.days
    agg["IsRepeatCustomer"] = (agg["TotalOrders"] > 1).astype(int)
    agg = agg.drop(columns=["FirstOrderDate", "LastOrderDate"])

    # Top-preference categorical columns (mode per customer), one-hot encoded
    for col, prefix in [("Product", "TopProduct"), ("PaymentMethod", "TopPayment"),
                        ("ReferralSource", "TopReferral")]:
        mode_series = grouped[col].agg(lambda s: s.mode().iloc[0])
        top_values = mode_series.value_counts().nlargest(TOP_N_CATEGORICAL).index
        mode_series = mode_series.where(mode_series.isin(top_values), other="Other")
        dummies = pd.get_dummies(mode_series, prefix=prefix)
        agg = agg.join(dummies)

    agg = agg.reset_index()
    return agg


def numeric_feature_columns(customer_df: pd.DataFrame) -> list[str]:
    """All columns eligible for scaling/PCA -- everything except the ID."""
    return [c for c in customer_df.columns if c != "CustomerID"]
