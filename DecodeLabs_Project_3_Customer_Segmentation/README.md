# Data Science — Project 3: Customer Segmentation (Unsupervised Learning)

**DecodeLabs Industrial Training Kit · Batch 2026**

Discovers hidden customer segments in unlabeled retail order data using PCA for
dimensionality reduction and K-Means clustering, with the optimal number of clusters
mathematically proven via the Elbow Method and Silhouette Score, translated into
actionable business personas.

> This repository was completed as part of a Data Science internship / training
> assignment at DecodeLabs. It builds on
> [Project 2](../DecodeLabs_Project2_Fraud_Detection)'s order-level output.

---

## ⚠️ Read this first: almost every customer has exactly one order

This dataset has **1,200 orders across ~1,189 unique customers** — nearly one order
per customer, with only ~0.9% of customers (11 people) placing more than one order.
Classic customer segmentation leans heavily on repeat-purchase history (frequency,
recency, loyalty), which barely exists here.

**A real failure mode was found and fixed during development, not hidden:** an early
version of this pipeline clustered on the full feature set, including
`TotalOrders`, `IsRepeatCustomer`, `TenureDays`, and `StdOrderValue`. Because those
columns are constant (or zero) for 99.1% of customers and only vary for the 11 repeat
buyers, `StandardScaler` turned that rare variation into enormous z-scores — which
then completely dominated PCA and K-Means, producing a trivial 99.1% / 0.9% split
that was really just "repeat buyer or not," not a meaningful segmentation. This is
demonstrated directly in the notebook (Section 2) before the fix is applied.

**The fix:** those 4 columns are excluded from the clustering feature set (they
remain in `data/processed/customer_features.csv` for reference) — see
`CLUSTERING_EXCLUDE_COLUMNS` in `src/pipeline.py`. Every cluster and persona in this
project should be read as a **single-transaction spending profile** (what a customer
bought, how much, how they paid) rather than a loyalty segment.

---

## Project Requirements → What Was Built

| Brief requirement | Implementation |
|---|---|
| Apply PCA to reduce 20+ columns into 2-3 dimensions | 33 features engineered per customer ([`src/customer_aggregation.py`](src/customer_aggregation.py)); PCA in [`src/dimensionality_reduction.py`](src/dimensionality_reduction.py) capped at 3 components |
| Use the Elbow Method and Silhouette Score to mathematically prove the optimal K | [`src/clustering.py`](src/clustering.py) — both computed for K=2..10; disagreement between them reported honestly (see Results) |
| Translate clusters into actionable business Personas | [`src/persona_builder.py`](src/persona_builder.py) — transparent, rule-based naming + a recommended action per persona |
| Key skills: dimensionality reduction (PCA), K-Means, distance metrics, business translation | Used throughout |

---

## Repository Structure

```
.
├── data/
│   ├── raw/
│   │   └── project2_orders_with_fraud_features.csv   # Project 2's order-level output (input here)
│   └── processed/
│       ├── customer_features.csv    # One row per customer, 33 engineered features (full set)
│       └── customer_segments.csv    # customer_features.csv + assigned Cluster
├── notebooks/
│   └── Customer_Segmentation.ipynb   # Narrative walkthrough incl. the trivial-split demo (pre-executed)
├── docs/
│   └── Project3_Customer_Segmentation_Documentation.pdf
├── reports/
│   ├── k_selection_diagnostics.csv    # WCSS + silhouette score per K
│   ├── clustering_summary.csv         # chosen K, final silhouette, PCA variance retained
│   ├── cluster_profile_summary.csv    # per-cluster means on the ORIGINAL feature scale
│   ├── customer_personas.csv          # persona name, size, %, recommended action
│   └── figures/
│       ├── 01_pca_explained_variance.png
│       ├── 02_elbow_and_silhouette.png
│       ├── 03_clusters_pca_2d.png
│       └── 04_persona_matrix.png
├── src/
│   ├── data_loader.py
│   ├── customer_aggregation.py
│   ├── dimensionality_reduction.py
│   ├── clustering.py
│   ├── persona_builder.py
│   ├── plotting.py
│   └── pipeline.py
├── tests/
│   └── test_pipeline.py              # 14 tests, several guarding the trivial-split fix specifically
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
pytest -v                       # runs all 14 tests
jupyter notebook notebooks/Customer_Segmentation.ipynb   # already pre-executed
```

---

## Methodology

### 1. Customer Aggregation (one row per customer)

33 features per customer, spanning spend behavior (`AvgOrderValue`, `TotalSpend`,
`AvgUnitPrice`, `AvgPricePerUnit`), engagement (`CouponUsageRate`, `WeekendOrderRate`,
`HighValueOrderRate`), risk (`FraudProxyRate`, from Project 2), variety
(`UniqueProducts`, `UniquePaymentMethods`), timing (`RecencyDays`, `AvgOrderMonth`,
`AvgOrderDayOfWeek`), and one-hot encoded top-5 product/payment/referral preferences.

### 2. Scale & Compress (StandardScaler → PCA)

All 33 features (minus the 4 excluded — see caveat above) are standardized, then PCA
finds the orthogonal axes of maximum variance. 22 components would be needed for a
95% cumulative-variance threshold; capped at 3 (per the brief) for clustering and
visualization, retaining ~26% of total variance. That's a real trade-off, stated
plainly rather than glossed over: with 29 candidate features contributing fairly
independently, there isn't much redundancy to compress without losing information.

### 3. Proving the Optimal K

Both the Elbow Method (WCSS via `kneed`'s `KneeLocator`) and the Silhouette Score are
computed for K = 2 through 10. **The two disagree** (Elbow: K=5, Silhouette: K=2) —
reported honestly rather than cherry-picked. Silhouette scores across the whole range
are modest (0.27–0.34), indicating this customer base sits more on a spending-level
continuum than in sharply distinct segments. The Silhouette Score is used as the
deciding metric (it directly measures cohesion/separation; the Elbow "knee" can be
ambiguous), giving a final **K = 2**.

### 4. Translating Clusters into Personas

K-Means runs in abstract PCA space, which means nothing to a marketing team.
Cluster assignments are mapped back onto the *original* customer features (not PCA
coordinates) to build each persona's profile, and a transparent rule (comparing each
cluster's `AvgOrderValue` and `CouponUsageRate` against the population mean) assigns
a name and a recommended action — deliberately readable so the reasoning behind each
label can be checked by hand against the summary table, not a black box.

---

## Results

| Persona | Size | % | Avg Order Value | Coupon Usage | Recommended Action |
|---|---|---|---|---|---|
| **Full-Price Big Spenders** | 441 | 37.1% | $1,907.70 | 74% | Prioritize for early access / premium loyalty perks. Weekend-heavy ordering — time promotions for Fri/Sat. |
| **Budget-Conscious Deal Hunters** | 748 | 62.9% | $552.59 | 75% | Run frequent flash sales / coupon campaigns — this segment converts primarily on discounts. |

Final silhouette score at K=2: **0.335** (a modest, not exceptional, separation —
consistent with the spending-continuum finding above, and far more trustworthy than
the misleadingly high 0.86 score the trivial "repeat-buyer" split produced before the
fix).

Full diagnostics in `reports/k_selection_diagnostics.csv` and
`reports/figures/02_elbow_and_silhouette.png`.

---

## Tech Stack

- **Python 3.12**
- **pandas** & **NumPy**
- **scikit-learn** — `StandardScaler`, `PCA`, `KMeans`, `silhouette_score`
- **kneed** — automated Elbow-point detection (`KneeLocator`)
- **matplotlib** & **seaborn** — diagnostics and persona visualizations
- **pytest** — unit testing
- **Jupyter / nbconvert** — the narrative notebook

See [`requirements.txt`](requirements.txt) for exact versions.

---

## Testing

14 unit tests, including several dedicated to the central lesson of this project:

- customer aggregation produces exactly one row per customer, with 20+ features
- single-order customers get `StdOrderValue = 0`, not `NaN`
- the 4 near-constant columns are verifiably excluded from the clustering feature set
- WCSS is monotonically non-increasing as K grows (a mathematical property of
  K-Means — if this ever fails, something is wrong with the fitting, not the data)
- a synthetic well-separated 2-blob dataset produces a high silhouette score (sanity
  check that the clustering code itself is correct)
- persona naming logic is verified against hand-constructed cluster profiles

```bash
pytest -v
```

---

## Author's Note

The most valuable finding in this project wasn't the final 2-persona split — it was
catching, diagnosing, and fixing the trivial 99%/1% split that a naive first pass
produced. That fix, and the reasoning behind it, is documented in code
(`CLUSTERING_EXCLUDE_COLUMNS`'s docstring), demonstrated directly in the notebook,
and guarded by unit tests — not just mentioned in passing. A segmentation result is
only useful if you can trust that it reflects real structure in the data rather than
an artifact of how a handful of outlier rows got scaled.

---

## License

Released under the [MIT License](LICENSE).
