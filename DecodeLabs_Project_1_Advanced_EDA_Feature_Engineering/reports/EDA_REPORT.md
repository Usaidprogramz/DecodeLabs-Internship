# EDA & Feature Engineering Report

## 1. Objective
Transform the supplied e-commerce order dataset into a mathematically clean, auditable, machine-learning-ready dataset while preserving source truth.

## 2. Raw Data Audit
- **Shape:** 1,200 rows x 14 columns.
- **Date coverage:** 2023-01-01 through 2025-06-30.
- **Primary identifiers:** `OrderID` is fully unique; `TrackingNumber` is also fully unique.
- **Missingness:** 309 values are missing in `CouponCode` (25.75%). No numeric field is missing.
- **Consistency:** `TotalPrice` matches `Quantity * UnitPrice` within cent-level tolerance throughout the source data.

## 3. Missing-Data Decision
The brief emphasizes statistical imputation rather than arbitrary guessing. The reusable pipeline therefore implements numeric **mean**, **median**, and **KNN** strategies. However, applying them to this exact dataset would be artificial because there are no missing numeric values.

`CouponCode` is categorical and its blank state has direct business meaning: the order did not use a coupon. Missing values are therefore represented as the explicit `NO_COUPON` category and summarized by the binary feature `HasCoupon`.

## 4. Outlier Analysis
IQR was evaluated on all numeric variables. `Quantity`, `UnitPrice`, and `ItemsInCart` contain no IQR outliers. `TotalPrice` has 8 values above the upper IQR bound of **3,330.41**.

These values are internally consistent with five-unit orders at high but valid unit prices. Deleting them would discard genuine transactions. The project therefore:
1. keeps the original `TotalPrice`,
2. adds `TotalPrice_IQR_Outlier`, and
3. creates `TotalPrice_Winsorized` clipped at the IQR boundaries for robust modeling.

This follows the brief's recommendation to cap rather than delete when row preservation matters.

## 5. Feature Engineering
The project engineers more than the required three predictive features:
- `HasCoupon`
- `OrderYear`
- `OrderMonth`
- `OrderQuarter`
- `OrderDayOfWeek`
- `IsWeekend`
- `BasketFillRatio`
- `OrderValuePerCartItem`
- `TotalPrice_IQR_Outlier`
- `TotalPrice_Winsorized`

Nominal dimensions (`Product`, `PaymentMethod`, `OrderStatus`, `ReferralSource`) are prepared for one-hot encoding to avoid the false distance relationships introduced by simple label encoding.

## 6. Vectorization
All transformation logic is expressed through Pandas/NumPy vectorized operations such as `.fillna`, `.clip`, `.div`, `.dt`, and matrix-based encoding. No row-by-row procedural preprocessing loop is required in the production pipeline.

## 7. Multicollinearity Audit
The raw numeric variables do not contain a Pearson correlation above 0.80. Engineered variables derived directly from source monetary fields can naturally become correlated; the pipeline provides `correlation_audit()` so the final modeling feature set can be pruned after the target variable is defined.

## 8. Structural Contracts
The pipeline asserts:
- unique `OrderID`,
- positive `UnitPrice` and `TotalPrice`,
- positive `ItemsInCart`,
- `Quantity` within the observed business range 1-5,
- `TotalPrice ≈ Quantity * UnitPrice`.

`src/schema_contract.py` adds an optional Pandera schema using lazy validation for production-style failure reporting.

## 9. Key Business Observations
- Total revenue: **1,264,761.96**.
- Average order value: **1,053.97**.
- Coupon usage: **74.25%**.
- Highest-revenue product: **Chair**, generating **195,620.11**.
- Order statuses are broadly distributed, with `Cancelled` (250) and `Returned` (247) the largest groups in this synthetic-style training dataset.

## 10. Production Extension
The brief discusses feature stores and point-in-time correctness. This internship task is a single batch dataset, so deploying Feast would add infrastructure without improving the submitted analysis. In a real predictive service, the engineered features should be registered once and reused by both offline training and online inference; historical joins must only use feature values available at each event timestamp to prevent leakage.

## 11. Conclusion
The final deliverable retains the original data for auditability, handles the real missingness semantically, provides the requested statistical-imputation machinery, isolates and neutralizes outliers without destructive row deletion, and creates a richer feature matrix suitable for downstream machine-learning experiments.
