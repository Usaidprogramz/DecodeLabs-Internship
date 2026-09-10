# Data Science — Project 2: Fraud Detection Pipeline

**DecodeLabs Industrial Training Kit · Batch 2026**

A leak-free supervised classification pipeline that flags potentially fraudulent orders using a **proxy target** in a highly imbalanced-by-design classification setup.

The project compares a linear baseline (**Logistic Regression**) against an ensemble model (**Random Forest**), with both models tuned using `GridSearchCV` to optimize **recall** and evaluated using **Precision, Recall, F1-Score, and ROC-AUC**.

**Accuracy is deliberately excluded.**

> This repository was completed as part of a Data Science internship / training assignment at DecodeLabs.
>
> It builds directly on **Project 1**, using its cleaned dataset as the starting point.

---

## ⚠️ Important: `IsFraud` Is a Proxy Label

The Project 1 orders dataset does **not** contain a genuine fraud/legitimate label.

To practice building a realistic fraud-detection classification pipeline, this project constructs a proxy target:

```text
IsFraud = 1  if OrderStatus is "Returned" or "Cancelled"

IsFraud = 0  otherwise
```

This means the model is **not detecting confirmed financial fraud**.

Instead, it is predicting **problem orders** represented by returned or cancelled orders.

This distinction is important because returned or cancelled orders can sometimes correlate with disputes, suspicious activity, or operational problems in real-world businesses, but they are **not automatically fraudulent**.

Therefore, throughout this project, the target is treated as a **fraud proxy** rather than genuine fraud.

The machine-learning architecture, however, follows the same principles that would be appropriate for a real fraud-detection problem:

* Leak-free train/test splitting
* Stratified sampling
* SMOTE for class imbalance
* Cross-validation
* Recall-first hyperparameter tuning
* Precision / Recall / F1 evaluation
* ROC-AUC evaluation
* Confusion matrices
* Feature-importance analysis

---

# 🎯 Project Requirements → Implementation

| Requirement                                                      | Implementation                                                   |
| ---------------------------------------------------------------- | ---------------------------------------------------------------- |
| Build a classification model for fraudulent/problem transactions | `IsFraud` proxy target created in `src/target_builder.py`        |
| Handle class imbalance using SMOTE                               | SMOTE implemented inside `imblearn.pipeline.Pipeline`            |
| Train multiple classification algorithms                         | Logistic Regression and Random Forest                            |
| Tune models using cross-validation                               | `GridSearchCV`                                                   |
| Optimize for recall                                              | `scoring="recall"`                                               |
| Avoid data leakage                                               | Stratified split before resampling and scaling                   |
| Evaluate without accuracy                                        | Precision, Recall, F1-Score and ROC-AUC                          |
| Analyze model behavior                                           | Confusion matrices, ROC curves, PR curves and feature importance |
| Test pipeline safety                                             | 17 automated tests                                               |

---

# 🛡️ Leak-Free Pipeline

This project specifically focuses on avoiding two major machine-learning problems.

### Trap #1 — The Illusion of Accuracy

Accuracy is **not calculated or reported** anywhere in the evaluation system.

Instead, the project focuses on:

* Precision
* Recall
* F1-Score
* ROC-AUC

This is particularly important for fraud-detection-style problems where missing a potentially problematic transaction can be more costly than generating a false alarm.

---

### Trap #2 — Data Leakage

SMOTE and feature scaling are applied **inside the machine-learning pipeline**, rather than before the train/test split.

The workflow is:

```text
Original Dataset
       ↓
Build IsFraud Proxy Target
       ↓
Feature Engineering
       ↓
Remove Leaky Features
       ↓
Stratified Train/Test Split
       ↓
       ├─────────────── Test Set
       │                   ↓
       │              Final Evaluation
       │
       └── Training Set
               ↓
        Cross Validation
               ↓
             SMOTE
               ↓
          Scaling*
               ↓
            Model
```

`*` Scaling is used for Logistic Regression but not for Random Forest.

Because SMOTE is contained inside `imblearn.pipeline.Pipeline`, synthetic samples are generated only from the training portion of each cross-validation fold.

The held-out validation/test data is never resampled.

---

# 📁 Repository Structure

```text
.
├── data/
│   ├── raw/
│   │   └── project1_cleaned_dataset.csv
│   │
│   └── processed/
│       └── orders_with_fraud_features.csv
│
├── notebooks/
│   └── Fraud_Detection_Pipeline.ipynb
│
├── reports/
│   ├── class_balance.csv
│   ├── model_evaluation_metrics.csv
│   ├── logistic_regression_best_params.csv
│   ├── random_forest_best_params.csv
│   ├── random_forest_feature_importance.csv
│   │
│   └── figures/
│       ├── 00_class_balance.png
│       └── 01_fraud_detection_dashboard.png
│
├── src/
│   ├── data_loader.py
│   ├── target_builder.py
│   ├── feature_engineering.py
│   ├── modeling.py
│   ├── evaluation.py
│   ├── plotting.py
│   └── pipeline.py
│
├── tests/
│   └── test_pipeline.py
│
├── main.py
├── requirements.txt
├── LICENSE
└── README.md
```

---

# 🚀 Quick Start

### 1. Create a virtual environment

```bash
python -m venv venv
```

### 2. Activate the environment

**Windows PowerShell:**

```powershell
venv\Scripts\activate
```

**Linux / macOS:**

```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the complete pipeline

```bash
python main.py
```

### 5. Run the tests

```bash
pytest -v
```

### 6. Open the notebook

```bash
jupyter notebook notebooks/Fraud_Detection_Pipeline.ipynb
```

The notebook contains the narrative walkthrough of the complete pipeline.

---

# 📊 Dataset

### Input Dataset

The project uses:

```text
data/raw/project1_cleaned_dataset.csv
```

This is the cleaned and feature-engineered output from **Project 1**.

The dataset contains:

* **1,200 orders**
* **25 original columns**
* No missing values
* Previously processed data from Project 1

Project 2 does not repeat the complete raw-data cleaning process because that work was already completed in Project 1.

---

# 🎯 Target Construction

The `IsFraud` proxy target is constructed using `OrderStatus`.

```python
IsFraud = 1
```

when:

```text
OrderStatus ∈ {Returned, Cancelled}
```

Otherwise:

```python
IsFraud = 0
```

### Actual Class Distribution

The pipeline produced the following class balance:

| Label           |     Count | Percentage |
| --------------- | --------: | ---------: |
| Legitimate (0)  |       703 |     58.58% |
| Fraud-proxy (1) |       497 |     41.42% |
| **Total**       | **1,200** |   **100%** |

---

## ⚠️ Class Imbalance Context

Although this project uses an imbalanced-learning workflow, this dataset is **not extremely imbalanced** compared with real-world fraud datasets.

The fraud-proxy class represents:

```text
497 / 1200 = 41.42%
```

This relatively high percentage occurs because the synthetic dataset assigns order statuses with approximately similar probabilities.

Therefore:

* `Returned + Cancelled` produces a large proxy-positive class.
* The dataset does not represent a realistic <1% fraud rate.
* SMOTE is still implemented to demonstrate the correct technique.
* Recall-first model tuning is still used to demonstrate a realistic fraud-detection workflow.

This limitation should be considered when interpreting the model results.

---

# 🧪 Methodology

## 1. Feature Engineering

Project 2 builds additional features on top of the cleaned Project 1 dataset.

Examples include:

| Feature         | Formula / Source               | Purpose                                |
| --------------- | ------------------------------ | -------------------------------------- |
| `AvgItemValue`  | `TotalPrice / ItemsInCart`     | Average value per cart item            |
| `ItemsPerOrder` | `ItemsInCart / (Quantity + 1)` | Captures cart-to-purchase relationship |
| `IsHighValue`   | `TotalPrice > 95th percentile` | Identifies unusually high-value orders |
| `PricePerUnit`  | `TotalPrice / Quantity`        | Normalizes order value by quantity     |
| `HasDiscount`   | `CouponUsed`                   | Indicates coupon usage                 |
| `WeekendFraud`  | `IsWeekendOrder × IsFraud`     | Exploratory analysis only              |

The resulting modeling dataset contains:

```text
37 features
```

before the train/test modeling process.

---

# 🔒 2. Leakage Prevention

Several columns are explicitly removed before model training.

### `OrderStatus`

`OrderStatus` is the source used to construct `IsFraud`.

Keeping it in the feature matrix would effectively give the model access to the answer.

Therefore:

```text
OrderStatus → DROPPED
```

---

### `WeekendFraud`

`WeekendFraud` directly contains the target:

```text
IsWeekendOrder × IsFraud
```

Therefore, it is used only for exploratory analysis and is never passed to the machine-learning models.

```text
WeekendFraud → DROPPED
```

---

### Identifiers

The following identifiers are also excluded because they do not provide meaningful generalizable predictive information:

```text
OrderID
CustomerID
TrackingNumber
ShippingAddress
```

Other raw/superseded columns are also removed where their information is already represented by engineered features.

---

# 3. Stratified Train/Test Split

The dataset is divided into:

```text
Training rows: 960
Test rows:     240
```

The split is performed using stratification.

```text
80% → Training
20% → Testing
```

The split occurs **before SMOTE or scaling**.

This ensures that:

* The test set remains untouched.
* The test set maintains the original class distribution.
* Synthetic SMOTE samples cannot leak into evaluation.

---

# 4. SMOTE

Because classification problems can contain unequal class distributions, the project uses:

```text
SMOTE
```

from `imbalanced-learn`.

SMOTE is implemented inside:

```text
imblearn.pipeline.Pipeline
```

rather than being applied manually to the complete dataset.

This is critical for preventing data leakage during cross-validation.

---

# 5. Logistic Regression Pipeline

The Logistic Regression pipeline is:

```text
StandardScaler
      ↓
SMOTE
      ↓
Logistic Regression
```

Scaling is important for Logistic Regression because the model is sensitive to feature magnitude and regularization.

### Best Hyperparameters

The final `GridSearchCV` search selected:

```text
C = 0.1
solver = liblinear
SMOTE k_neighbors = 7
```

Best cross-validation recall:

```text
0.507
```

---

# 6. Random Forest Pipeline

The Random Forest pipeline is:

```text
SMOTE
   ↓
Random Forest
```

A scaler is not required because tree-based models are generally insensitive to feature scale.

### Best Hyperparameters

The final `GridSearchCV` search selected:

```text
class_weight = balanced
max_depth = None
n_estimators = 200
SMOTE k_neighbors = 5
```

Best cross-validation recall:

```text
0.274
```

---

# 7. Hyperparameter Tuning

Both models are tuned using:

```text
GridSearchCV
```

The optimization metric is:

```text
Recall
```

rather than accuracy.

This reflects the project's focus on identifying as many potentially problematic orders as possible.

---

# 📈 Results

The final test-set evaluation produced:

| Metric    | Logistic Regression | Random Forest |
| --------- | ------------------: | ------------: |
| Precision |           **0.377** |         0.344 |
| Recall    |           **0.434** |         0.222 |
| F1-Score  |           **0.404** |         0.270 |
| ROC-AUC   |               0.437 |     **0.442** |

### Best Model by Recall

**Logistic Regression**

```text
Recall = 0.434
```

Logistic Regression was selected as the best model according to the project's primary selection criterion: **recall**.

---

# 🔍 Results Interpretation

The results should be interpreted honestly.

Neither model demonstrates strong predictive performance on the test set.

The ROC-AUC scores are:

```text
Logistic Regression → 0.437
Random Forest       → 0.442
```

Both are below the 0.50 level associated with random ranking.

This suggests that the available features contain **very limited predictive signal for the `IsFraud` proxy target**.

This is not necessarily a pipeline failure.

The synthetic dataset's `OrderStatus` does not appear to have a strong relationship with the other available order attributes. Since `OrderStatus` is also the source of the target, it must be removed from the model to prevent leakage.

The result therefore demonstrates an important machine-learning principle:

> A correctly implemented model cannot manufacture meaningful predictive signal when the available features do not contain it.

A leaky pipeline could have produced artificially impressive results by allowing `OrderStatus` into the feature matrix or by applying SMOTE before the train/test split.

This project deliberately avoids those practices.

---

# 🌲 Random Forest Feature Importance

The most important Random Forest features were:

1. `TotalPrice`
2. `PricePerUnit`
3. `AvgItemValue`
4. `UnitPrice`
5. `OrderMonth`

The complete ranking is available in:

```text
reports/random_forest_feature_importance.csv
```

Feature importance should be interpreted cautiously because the overall ROC-AUC indicates that these features provide limited predictive power for the proxy target.

---

# 📊 Generated Reports

Running:

```bash
python main.py
```

generates the following outputs:

```text
data/processed/orders_with_fraud_features.csv

reports/class_balance.csv

reports/model_evaluation_metrics.csv

reports/logistic_regression_best_params.csv

reports/random_forest_best_params.csv

reports/random_forest_feature_importance.csv

reports/figures/00_class_balance.png

reports/figures/01_fraud_detection_dashboard.png
```

The dashboard contains visualizations including:

* Confusion matrices
* ROC curves
* Precision-Recall curves
* Model metric comparison
* Random Forest feature importance

---

# 🧪 Testing

The project contains:

```text
17 automated tests
```

Run them with:

```bash
pytest -v
```

The test suite validates important properties of the pipeline, including:

* `OrderStatus` is removed before modeling.
* `WeekendFraud` is removed before modeling.
* Leaky columns are not present in the feature matrix.
* The final feature matrix is numeric.
* The train/test split occurs before resampling.
* SMOTE does not modify the held-out test set.
* Stratification preserves the class distribution.
* The modeling pipeline can be fitted successfully.
* `EvaluationResult` does not contain an accuracy field.

The leakage-safety tests are particularly important because preventing data leakage is one of the main goals of this project.

---

# 🧰 Tech Stack

### Programming

* Python 3.12

### Data Processing

* pandas
* NumPy

### Machine Learning

* scikit-learn
* Logistic Regression
* Random Forest
* GridSearchCV
* Cross-validation
* Feature scaling

### Imbalanced Learning

* imbalanced-learn
* SMOTE
* `imblearn.pipeline.Pipeline`

### Evaluation

* Precision
* Recall
* F1-Score
* ROC-AUC
* Confusion Matrix
* ROC Curve
* Precision-Recall Curve

### Visualization

* Matplotlib
* Seaborn

### Testing

* pytest

### Development / Documentation

* Jupyter Notebook
* nbconvert
* VS Code
* Git
* GitHub

---

# 📚 Key Concepts Demonstrated

This project demonstrates practical understanding of:

* Binary classification
* Logistic Regression
* Random Forest
* Imbalanced datasets
* SMOTE
* Stratified train/test splitting
* Cross-validation
* GridSearchCV
* Hyperparameter tuning
* Feature engineering
* Feature scaling
* Data leakage prevention
* Precision vs Recall
* F1-Score
* ROC-AUC
* Confusion matrices
* Feature importance
* Unit testing
* Reproducible machine-learning pipelines

---

# 💡 Key Lessons

### 1. Accuracy is not always the right metric

For fraud-detection-style problems, accuracy can hide poor minority-class performance.

This project therefore focuses on:

```text
Precision
Recall
F1
ROC-AUC
```

---

### 2. Data leakage can create misleading results

Using the target itself as an input feature can make a model appear extremely accurate while making it useless on unseen data.

This project explicitly removes target-derived features before modeling.

---

### 3. SMOTE must be used correctly

SMOTE should not be applied to the entire dataset before splitting.

Instead:

```text
Training Fold
     ↓
SMOTE
     ↓
Model
```

The validation/test data remains untouched.

---

### 4. Good pipelines can still produce poor scores

A low score does not automatically mean the implementation is wrong.

If the available features contain little predictive information, a properly designed model should report weak performance rather than artificially inflate its results.

---

# 📝 Author's Note

This project intentionally documents the limitations of the dataset rather than presenting the proxy target as genuine fraud.

The `IsFraud` label is derived from:

```text
Returned + Cancelled orders
```

and therefore should be interpreted only as a **problem-order / fraud-proxy classification task**.

The near-random ROC-AUC results are reported exactly as produced by the pipeline.

No accuracy score has been added to make the results appear better, and no leaky features have been retained to artificially increase model performance.

The primary objective of this project is therefore not simply to achieve a high score, but to demonstrate how to build a **reproducible, leak-free, recall-focused classification pipeline** and honestly evaluate its limitations.

---

# 🔗 Related Project

This project builds on the cleaned and feature-engineered dataset produced in **Project 1**.

```text
Project 1
Advanced EDA & Feature Engineering
        ↓
Cleaned Dataset
        ↓
Project 2
Fraud Detection Pipeline
```

---

# 📄 License

This project is released under the [MIT License](LICENSE).

---

## ⭐ Project Summary

```text
Dataset
   ↓
Target Construction
   ↓
Feature Engineering
   ↓
Leakage Prevention
   ↓
Stratified Train/Test Split
   ↓
SMOTE
   ↓
Logistic Regression + Random Forest
   ↓
GridSearchCV
   ↓
Recall Optimization
   ↓
Precision / Recall / F1 / ROC-AUC
   ↓
Final Evaluation & Analysis
```

**Built as part of the DecodeLabs Data Science Industrial Training Program — Batch 2026.**
