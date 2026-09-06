# Data Science — Project 1: Advanced EDA & Feature Engineering

**DecodeLabs Industrial Training Kit · Batch 2026**

An end-to-end, production-style data cleaning and feature engineering pipeline built for
an e-commerce orders dataset (1,200 rows / 14 columns). This project applies the
brief's core requirement — *mathematical clarity before modeling* — through statistical
imputation, IQR-based outlier neutralization, engineered predictive features, and a
runtime schema contract, all backed by unit tests and a fully reproducible notebook.

> This repository was completed as part of a Data Science internship / training
> assignment at DecodeLabs.

---

## Project Requirements → What Was Built

| Brief requirement | Implementation |
|---|---|
| Handle missing data via statistical imputation (Mean / Median / KNN) | [`src/missing_value_handler.py`](src/missing_value_handler.py) implements the full <5% / 5–20% / >20% decision matrix, **plus** a documented, deliberate override for `CouponCode` (see below). A dedicated controlled experiment comparing **Mean vs. Median vs. KNN** lives in [`src/imputation_comparison_demo.py`](src/imputation_comparison_demo.py). |
| Identify and neutralize outliers using Z-Scores or IQR | [`src/outlier_handler.py`](src/outlier_handler.py) — IQR fences (`Q1 - 1.5×IQR`, `Q3 + 1.5×IQR`), neutralized by **winsorization** (`numpy.clip`) rather than row deletion, with `_was_outlier` flag columns so the signal isn't lost. |
| Engineer at least 3 new predictive features | [`src/feature_engineering.py`](src/feature_engineering.py) — **7** engineered features (see table below). |
| Key skills: Pandas, NumPy, statistical analysis, data cleaning, feature extraction | Used throughout; see `requirements.txt`. |

The brief's later slides (Enterprise-Grade Data Engineering) also describe an
Input → Process → Output architecture, runtime data contracts (Pandera-style), and a
feature-store concept. This project mirrors that structure at a scope appropriate for
a single dataset:

- **Input** — [`src/data_loader.py`](src/data_loader.py), [`src/missing_value_handler.py`](src/missing_value_handler.py), [`src/outlier_handler.py`](src/outlier_handler.py)
- **Process** — [`src/feature_engineering.py`](src/feature_engineering.py) (feature creation, one-hot encoding, multicollinearity check), [`src/scaler.py`](src/scaler.py) (StandardScaler on continuous numeric features)
- **Output** — [`src/schema_validator.py`](src/schema_validator.py), a dependency-free, lazily-evaluated schema contract inspired by Pandera's `lazy=True` behavior (collects *every* validation failure into one report instead of stopping at the first)

Everything is orchestrated by [`src/pipeline.py`](src/pipeline.py) and run with a single
`python main.py`.

---

## Repository Structure

```
.
├── data/
│   ├── raw/                             # Original, untouched input
│   │   └── Dataset_for_Data_Analytics.xlsx
│   └── processed/                       # Pipeline outputs (generated)
│       ├── cleaned_dataset.csv          # Cleaned + feature-engineered, human-readable
│       ├── ml_ready_encoded_dataset.csv # One-hot encoded, unscaled, ready for sklearn
│       └── ml_ready_scaled_dataset.csv  # One-hot encoded + StandardScaler applied
├── notebooks/
│   └── EDA_and_Feature_Engineering.ipynb  # Narrative walkthrough with charts (pre-executed)
├── reports/                             # Audit trail (generated)
│   ├── raw_data_overview.csv
│   ├── imputation_log.csv
│   ├── imputation_strategy_comparison.csv
│   ├── outlier_report.csv
│   ├── multicollinearity_report.csv
│   ├── schema_validation_failures.csv
│   ├── scaling_log.csv
│   └── figures/                         # Saved PNG charts
├── src/
│   ├── data_loader.py
│   ├── missing_value_handler.py
│   ├── outlier_handler.py
│   ├── feature_engineering.py
│   ├── scaler.py
│   ├── schema_validator.py
│   ├── imputation_comparison_demo.py
│   └── pipeline.py
├── tests/
│   └── test_pipeline.py                 # 17 unit tests, one per pipeline behavior
├── main.py                              # Single entry point
├── requirements.txt
├── LICENSE                              # MIT
└── README.md
```

---

## Quick Start

```bash
# 1. Clone / unzip the project, then from the project root:
python -m venv venv
source venv/bin/activate        # on Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the full pipeline
python main.py

# 4. Run the Mean vs. Median vs. KNN imputation comparison
python -m src.imputation_comparison_demo

# 5. Run the test suite
pytest -v

# 6. (Optional) Open the notebook — it's already pre-executed with saved outputs
jupyter notebook notebooks/EDA_and_Feature_Engineering.ipynb
```

Running `main.py` regenerates every file under `data/processed/` and `reports/` from
the raw Excel file in `data/raw/`, so the whole project is reproducible from a single
command.

---

## Dataset

`data/raw/Dataset_for_Data_Analytics.xlsx` — 1,200 e-commerce order line items, 14
columns: `OrderID`, `Date`, `CustomerID`, `Product`, `Quantity`, `UnitPrice`,
`ShippingAddress`, `PaymentMethod`, `OrderStatus`, `TrackingNumber`, `ItemsInCart`,
`CouponCode`, `ReferralSource`, `TotalPrice`.

**Missingness:** every column is fully populated except `CouponCode`, which is missing
in 309 / 1,200 rows (**25.75%**).

---

## 1. Missing Data Handling

The brief specifies a missingness-proportion decision matrix:

| Missingness | Strategy |
|---|---|
| < 5% | Drop rows |
| 5–20% | Statistical imputation (median for skewed numeric; sub-group mode for categorical) |
| > 20% | Multi-dimensional estimation (KNN) |

`CouponCode`'s 25.75% missingness falls in the ">20% → KNN" bucket by that rule alone.
**This project deliberately overrides that rule for this column**, and logs the reason
in `reports/imputation_log.csv`:

> A blank coupon code almost certainly means *no coupon was applied at checkout* — the
> missingness is informative (Missing **Not** At Random), not a random data-entry gap.
> Estimating a plausible-looking code with KNN would invent information that never
> existed. Instead: a `CouponUsed` binary flag is engineered from the **original** null
> pattern first, and only then is the text column filled with the domain-meaningful
> sentinel `"NoCoupon"`.

Since this dataset's *numeric* columns have 0% real missingness, there was nothing
genuine to compare Mean vs. Median vs. KNN on. To still address that requirement
properly, [`src/imputation_comparison_demo.py`](src/imputation_comparison_demo.py) runs
a **controlled experiment**: it punches a 15% Missing-Completely-At-Random hole into
`UnitPrice` (so the true values are still known for scoring), imputes it three ways, and
measures reconstruction error and distribution distortion against the ground truth:

| Strategy | RMSE on masked values | Std. dev. distortion |
|---|---|---|
| **KNN (k=5)** | **lowest** | **~1–2%** |
| Mean | higher | ~7–8% |
| Median | higher | ~7–8% |

**Finding:** KNN reconstructs the true distribution shape far more faithfully than Mean
or Median, which both flatten variance because every missing value collapses to the
same constant. This is the exact trade-off the brief describes — mean/median imputation
is cheap and stable but "artificially deflates standard deviation," while KNN "captures
complex multi-dimensional relationships" at higher computational cost. See
`reports/figures/07_imputation_strategy_comparison.png` for the visual comparison.

---

## 2. Outlier Detection & Neutralization

Outliers are detected with the IQR fence on every numeric column
(`Quantity`, `UnitPrice`, `ItemsInCart`, `TotalPrice`):

```
Lower Bound = Q1 − 1.5 × IQR
Upper Bound = Q3 + 1.5 × IQR
```

Only `TotalPrice` has genuine outliers — **8 rows (0.67%)**, large bulk orders beyond
the upper fence. Rather than deleting these rows (which would destroy the paired
`Quantity` / `UnitPrice` / `Product` values for that order), values are **winsorized**
(capped exactly at the fence with `numpy.clip`), and a `TotalPrice_was_outlier` flag
column is added so a downstream model can still learn "this was an extreme value" as a
signal instead of it being silently erased. Full numbers in
`reports/outlier_report.csv`; before/after boxplots in `reports/figures/`.

---

## 3. Engineered Features

| Feature | Formula / Source | Why it matters |
|---|---|---|
| `OrderMonth` | `Date.month` | Captures monthly seasonality in retail demand |
| `OrderDayOfWeek` | `Date.dayofweek` | Weekday purchase patterns differ from weekends |
| `IsWeekendOrder` | `OrderDayOfWeek ∈ {5, 6}` | Simple binary seasonality flag |
| `CouponUsed` | `CouponCode.notna()` (captured *before* sentinel fill) | Preserves the informative missingness signal from `CouponCode` |
| `CartFillRatio` | `Quantity / ItemsInCart` | Proxy for purchase decisiveness / cart-abandonment risk |
| `CustomerOrderCount` | `CustomerID` value counts | Frequency-encodes customer identity |
| `IsRepeatCustomer` | `CustomerOrderCount > 1` | Repeat customers have different LTV / return-rate behavior than first-timers |

(7 features — more than double the brief's minimum of 3.)

---

## 4. Categorical Encoding & Multicollinearity

Nominal columns (`Product`, `PaymentMethod`, `OrderStatus`, `ReferralSource`,
`CouponCode`) are **one-hot encoded**, not label/integer-encoded, so no false ordinal
distance is introduced between categories with no natural order (e.g. "Tokyo" is not
mathematically "3× London").

Before finalizing, the numeric feature matrix is checked for multicollinearity
(`|correlation| > 0.80`). None was found in this dataset (`reports/multicollinearity_report.csv`
is empty), but `src/feature_engineering.resolve_multicollinearity()` implements the
general fix: for any collinear pair, it keeps whichever feature correlates more
strongly with the target variable and drops the weaker one — rather than arbitrarily
dropping the first column encountered.

---

## 5. Feature Scaling (StandardScaler)

The final step in Phase 2 standardizes continuous numeric features to zero mean / unit
variance with scikit-learn's `StandardScaler` ([`src/scaler.py`](src/scaler.py)). This
matters for any distance-based or gradient-based estimator (KNN, SVM, logistic
regression, neural nets) — without it, a feature like `TotalPrice` (scale of hundreds to
thousands) would dominate a feature like `CartFillRatio` (scale of 0–1) purely because
of its units, not because it's more predictive.

Two design choices worth calling out:

- **Scaling runs *after* winsorization, not before.** `StandardScaler` fits on mean and
  standard deviation, both of which are themselves sensitive to outliers. Scaling on the
  raw (pre-winsorized) data would let a handful of extreme values distort the scale for
  every other row.
- **Only continuous columns are scaled** — `Quantity`, `UnitPrice`, `ItemsInCart`,
  `TotalPrice`, `CartFillRatio`, `CustomerOrderCount`, `OrderMonth`, `OrderDayOfWeek`.
  Binary flags (`IsWeekendOrder`, `CouponUsed`, `IsRepeatCustomer`, every
  `*_was_outlier` column) and one-hot encoded columns are left untouched — scaling a 0/1
  indicator doesn't help most estimators and only makes the column harder to read as a
  plain flag.

Output: `data/processed/ml_ready_scaled_dataset.csv` (a separate file from the unscaled
`ml_ready_encoded_dataset.csv`, so the human-interpretable version is still available
alongside the model-ready one). Pre-scale mean/std for every column is logged to
`reports/scaling_log.csv`.

---

## 6. Output Contract — Schema Validation

`src/schema_validator.py` is a lightweight, dependency-free "data contract" checker
inspired by Pandera's runtime schema validation. Every column in the final cleaned
dataset is checked against its expected dtype, nullability, and value bounds. Validation
is **lazy** — like Pandera's `lazy=True` — meaning it does not stop at the first
failure; every issue found is collected into a single `ValidationResult` and written to
`reports/schema_validation_failures.csv`, so the whole dataset can be diagnosed in one
pass instead of a crash-fix-rerun loop.

---

## Key Business Insights

Beyond the engineering pipeline, here's what the cleaned dataset actually says about the
underlying orders (computed directly from `data/processed/cleaned_dataset.csv`):

- **Product mix is broad, not top-heavy** — the 7 products are nearly evenly split
  (13.0%–15.1% of orders each); `Printer` narrowly leads at 15.1%.
- **Payment methods are evenly distributed** — `Online` is the most-used method at
  21.5%, with `Cash`, `Credit Card`, `Debit Card`, and `Gift Card` all within a few
  points of each other (19.2%–20.5%).
- **Order outcomes skew slightly unfavorable** — `Cancelled` (20.8%) and `Returned`
  (20.6%) are together more common than `Delivered` (19.2%), worth flagging for a
  fulfillment/quality investigation.
- **Coupon usage is high** — 74.2% of orders had a coupon code applied at checkout.
- **Weekday orders dominate** — only 29.8% of orders happen on a weekend; the large
  majority of purchases occur Monday–Friday.
- **Repeat customers are rare in this sample** — only 1.8% of customers placed more than
  one order, meaning most modeling signal will come from first-time-buyer behavior
  rather than loyalty patterns.
- **Average order value is $1,053.64**, with an average cart-fill ratio of 0.58 (a
  typical order captures about 58% of the items sitting in the customer's cart).

*(These are read directly off the cleaned dataset for this specific run; regenerate with
`python main.py` and inspect `data/processed/cleaned_dataset.csv` to reproduce or update
them against a refreshed data pull.)*

---

## Results Summary

| Metric | Before | After |
|---|---|---|
| Rows | 1,200 | 1,200 *(no rows dropped)* |
| Columns | 14 | 25 *(+11: 7 engineered features + 4 outlier flag columns)* |
| Missing values | 309 (`CouponCode`) | 0 |
| Outliers (IQR, all numeric cols) | 8 (`TotalPrice`) | 0 (winsorized, flagged) |
| Multicollinear feature pairs (\|r\| > 0.80) | — | 0 |
| Numeric features standardized (StandardScaler) | — | 8 |
| Schema validation | — | **Passed** |

---

## Tech Stack

- **Python 3.12**
- **pandas** & **NumPy** — data manipulation, vectorized statistics
- **scikit-learn** — `KNNImputer`, `StandardScaler`, error metrics
- **matplotlib** & **seaborn** — visualization
- **pytest** — unit testing
- **Jupyter / nbconvert** — the narrative notebook

See [`requirements.txt`](requirements.txt) for exact versions.

---

## Testing

17 unit tests cover every stage of the pipeline (missing-value routing at each
threshold band, IQR outlier detection, winsorization without row loss, each engineered
feature, one-hot encoding, multicollinearity detection, StandardScaler correctness, and
schema validation — both passing and failing cases):

```bash
pytest -v
```

---

**Author:** Muhammad Usaid — DecodeLabs Data Science Intern, Batch 2026
## Author's Note

This project deliberately treats data cleaning as engineering, not busywork: every
imputation and outlier decision is logged with a stated reason (see
`reports/imputation_log.csv`), and the one place the generic rule from the brief was
overridden (`CouponCode`) is explained rather than silently applied, in line with the
brief's own framing: *"data preprocessing is not janitorial work; it is the structural
engineering of mathematical truth."*

---

## License

Released under the [MIT License](LICENSE).
