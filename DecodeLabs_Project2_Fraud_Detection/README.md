# Data Science — Project 2: Fraud Detection Pipeline

**DecodeLabs Industrial Training Kit · Batch 2026**

A leak-free supervised classification pipeline that flags potentially fraudulent
orders in a highly-imbalanced-by-design setup, comparing a linear baseline
(Logistic Regression) against an ensemble model (Random Forest), tuned with
`GridSearchCV` on recall and evaluated with Precision/Recall/F1/ROC-AUC —
**accuracy deliberately excluded**.

> This repository was completed as part of a Data Science internship / training
> assignment at DecodeLabs. It builds directly on
> [Project 1](../DecodeLabs_Project_1_Advanced_EDA_Feature_Engineering) — see the
> **Dataset** section below for exactly what's reused and why.

---

## ⚠️ Read this first: `IsFraud` is a proxy label, not real fraud data

The Project 1 orders dataset has **no genuine fraud/legitimate column**. To still
build and practice a real fraud-detection *pipeline* on it, this project defines:

```
IsFraud = 1   if OrderStatus is "Returned" or "Cancelled"
IsFraud = 0   otherwise
```

This is a legitimate, common technique when true labels aren't available — problem
orders (returns, cancellations) often correlate with disputes or bad-actor behavior
in real businesses. **But it is a simplification, not a claim that these orders are
confirmed fraud**, and every result in this project is described as "predicting
problem orders (returned/cancelled)," never as "detecting confirmed financial
fraud." The modeling architecture (leak-free splits, SMOTE, recall-first tuning,
Precision/Recall/ROC-AUC evaluation) is exactly what a real fraud pipeline would
use — the label is the only thing simplified here.

---

## Project Requirements → What Was Built

| Brief requirement | Implementation |
|---|---|
| Build a classification model to identify fraudulent transactions in a highly imbalanced dataset | `IsFraud` proxy target built in [`src/target_builder.py`](src/target_builder.py); full pipeline in [`src/pipeline.py`](src/pipeline.py) |
| Implement SMOTE to handle class imbalance | [`src/modeling.py`](src/modeling.py) — SMOTE wrapped inside `imblearn.pipeline.Pipeline`, re-fit on every CV training fold only |
| Train multiple algorithms (Logistic Regression, Random Forest) using Scikit-Learn | Both implemented, tuned via `GridSearchCV` |
| Discard "Accuracy" and evaluate using strict Precision, Recall, and ROC-AUC | [`src/evaluation.py`](src/evaluation.py) — `EvaluationResult` has no accuracy field at all (tested) |
| Key skills: classification algorithms, Scikit-Learn pipelines, imbalanced data handling, hyperparameter tuning | Used throughout |

The brief's slide deck ("The Leak-Free Pipeline") calls out two specific traps this
project is built to avoid:

- **Trap #1 — The Illusion of Accuracy:** never computed here. See `evaluation.py`.
- **Trap #2 — The Data Leakage Catastrophe:** SMOTE and scaling are applied **only**
  inside `imblearn.pipeline.Pipeline`, and the train/test split happens **before**
  either ever runs. See the Methodology section below and
  `tests/test_pipeline.py`'s leakage-safety tests.

---

## Repository Structure

```
.
├── data/
│   ├── raw/
│   │   └── project1_cleaned_dataset.csv   # Project 1's cleaned output (input here)
│   └── processed/
│       └── orders_with_fraud_features.csv # + IsFraud target + 6 engineered features
├── notebooks/
│   └── Fraud_Detection_Pipeline.ipynb     # Narrative walkthrough (pre-executed)
├── reports/
│   ├── class_balance.csv
│   ├── model_evaluation_metrics.csv
│   ├── logistic_regression_best_params.csv
│   ├── random_forest_best_params.csv
│   ├── random_forest_feature_importance.csv
│   └── figures/
│       ├── 00_class_balance.png
│       └── 01_fraud_detection_dashboard.png
├── src/
│   ├── data_loader.py
│   ├── target_builder.py
│   ├── feature_engineering.py
│   ├── modeling.py
│   ├── evaluation.py
│   ├── plotting.py
│   └── pipeline.py
├── tests/
│   └── test_pipeline.py                   # 17 tests, ~half dedicated to leakage safety
├── main.py
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Quick Start

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt

python main.py                  # runs the full pipeline end-to-end
pytest -v                       # runs all 17 tests, including leakage-safety checks
jupyter notebook notebooks/Fraud_Detection_Pipeline.ipynb   # already pre-executed
```

`main.py` regenerates every file under `data/processed/` and `reports/` from
`data/raw/project1_cleaned_dataset.csv`.

---

## Dataset

**Input:** `data/raw/project1_cleaned_dataset.csv` — the fully cleaned,
feature-engineered output of Project 1 (1,200 orders, 25 columns; no missing
values, outliers already winsorized). This project does not re-clean raw data —
Project 1 already did that work, and Project 2 builds directly on top of it.

**Target construction:** `IsFraud` = 1 for `OrderStatus` in `{Returned, Cancelled}`,
else 0. On this dataset that comes out to:

| Label | Count | % |
|---|---|---|
| Legitimate (0) | 703 | 58.6% |
| Fraud-proxy (1) | 497 | 41.4% |

**Worth noting honestly:** this is nowhere near the extreme (<1%) imbalance real
fraud datasets have — it's a direct consequence of the synthetic data generator
assigning each of the 5 order statuses roughly equal probability (~20% each), so
`Returned + Cancelled ≈ 40%`. The full imbalanced-learning toolkit (SMOTE,
recall-first tuning) is still applied exactly as a genuinely rare-event problem
would require, but the lift from these techniques is naturally more modest here
than on a truly imbalanced dataset.

---

## Methodology

### 1. Feature Engineering

Six new features, built on top of Project 1's cleaned columns:

| Feature | Formula | Purpose |
|---|---|---|
| `AvgItemValue` | `TotalPrice / ItemsInCart` | Average value per item in the cart |
| `ItemsPerOrder` | `ItemsInCart / (Quantity + 1)` | Cart-to-purchase ratio |
| `IsHighValue` | `TotalPrice` > 95th percentile | Flags unusually large orders |
| `PricePerUnit` | `TotalPrice / Quantity` | Normalizes price by quantity |
| `HasDiscount` | `CouponUsed` (from Project 1) | Coupon usage as a candidate signal |
| `WeekendFraud` | `IsWeekendOrder × IsFraud` | **Exploratory only** — see below |

### 2. Leakage Guardrails

Two categories of columns are explicitly dropped before modeling
(`src/feature_engineering.drop_leaky_columns`), each with a dedicated test:

- **`OrderStatus`** — the literal source of the label. Leaving it in the feature
  matrix would let the model "predict" fraud by reading the answer key.
- **`WeekendFraud`** — a direct function of the label (`IsWeekendOrder * IsFraud`),
  built only for the exploratory weekday-vs-weekend chart in the notebook. Never a
  model input.
- **Identifiers** (`OrderID`, `CustomerID`, `TrackingNumber`, `ShippingAddress`) and
  **already-encoded originals** (raw `Date`, raw `CouponCode`) — no generalizable
  signal, or superseded by an already-engineered column.

### 3. Stratified Split *Before* Any Resampling

An 80/20 **stratified** train/test split runs first, before SMOTE or scaling touch
anything. This guarantees the test set reflects the *true* class distribution and
that no synthetic information can leak into evaluation.

### 4. Leak-Free SMOTE via `imblearn.pipeline.Pipeline`

A plain `sklearn.pipeline.Pipeline` only knows how to transform `X`. SMOTE needs to
change **both** `X` and `y` (it invents new minority-class rows), so it doesn't fit
that interface. `imblearn.pipeline.Pipeline` supports `fit_resample()` and, by
construction, only ever applies it to whatever data it receives at `.fit()` time —
which inside `GridSearchCV`'s cross-validation is only each fold's *training* split.
The held-out fold in every CV round is never resampled.

- **Logistic Regression:** `StandardScaler → SMOTE → LogisticRegression`
  (unscaled features would distort the regularization penalty)
- **Random Forest:** `SMOTE → RandomForestClassifier` (no scaler — tree splits are
  scale-invariant)

Both are tuned with `GridSearchCV` optimizing **recall**, not accuracy: in fraud
detection, a missed fraud case (false negative) is typically far more costly than a
false alarm.

### 5. Evaluation — Accuracy Deliberately Excluded

`EvaluationResult` in `src/evaluation.py` has no accuracy field at all — not
"computed but ignored," genuinely not present, and a unit test asserts it stays
that way. Precision, Recall, F1, and ROC-AUC are computed instead, alongside
confusion matrices, ROC curves, precision-recall curves, and Random Forest feature
importances.

---

## Results

| Metric | Logistic Regression | Random Forest |
|---|---|---|
| Precision | 0.377 | 0.368 |
| **Recall** | **0.434** | 0.253 |
| F1-Score | 0.404 | 0.299 |
| ROC-AUC | 0.437 | 0.469 |

**Best model by recall (the brief's selection criterion): Logistic Regression.**

**Honest interpretation:** both models land close to **0.44–0.47 ROC-AUC** —
essentially indistinguishable from a random classifier (AUC 0.50). This is a real
finding, not a pipeline bug: the synthetic dataset assigns `OrderStatus` with no
apparent causal relationship to any of the order's other attributes, so the
`IsFraud` proxy has no real signal for these models to learn from the available
features. The pipeline is working correctly — there's simply nothing predictive to
find in this particular synthetic dataset. This is precisely why the leak-free
discipline in this project matters: a leaky pipeline (SMOTE before splitting, or
`OrderStatus` left in the features) could easily have produced a deceptively good
looking score that wouldn't hold up on genuinely new data. The honest, near-random
result here is evidence the pipeline can be trusted, not a discouraging outcome to
paper over.

Top Random Forest predictive signals (by importance): `TotalPrice`, `PricePerUnit`,
`AvgItemValue`, `UnitPrice`, `OrderMonth` — see
`reports/random_forest_feature_importance.csv` for the full ranking.

Full dashboard (confusion matrices, ROC curves, PR curves, feature importance,
metric comparison): `reports/figures/01_fraud_detection_dashboard.png`.

---

## Tech Stack

- **Python 3.12**
- **pandas** & **NumPy**
- **scikit-learn** — classification, `GridSearchCV`, metrics
- **imbalanced-learn** — `SMOTE`, `imblearn.pipeline.Pipeline`
- **matplotlib** & **seaborn** — dashboard visualization
- **pytest** — unit testing
- **Jupyter / nbconvert** — the narrative notebook

See [`requirements.txt`](requirements.txt) for exact versions.

---

## Testing

17 unit tests, roughly half dedicated specifically to **leakage safety** — the
central risk this brief warns about:

- `OrderStatus` and `WeekendFraud` never enter the feature matrix (explicit
  assertions on the actual column list)
- every column named in the leaky-columns drop list is verifiably gone afterward
- the feature matrix is fully numeric
- a fitted SMOTE pipeline never resamples the test set
- a stratified split preserves the true (imbalanced) class ratio in the held-out set
- `EvaluationResult` has no accuracy field, by construction

```bash
pytest -v
```

---

## Author's Note

This project treats the absence of a real fraud label as something to state
plainly rather than paper over — see the caveat at the top of this README and in
the notebook. The near-random ROC-AUC in the results section is reported exactly
as it came out of the pipeline, with the reason explained, rather than smoothed
into a more flattering number. A pipeline's leak-free discipline is only worth
something if it's trusted to report an honest result even when that result isn't
exciting — that's the standard this project holds itself to.

---

## License

Released under the [MIT License](LICENSE).
