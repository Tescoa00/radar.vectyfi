# Vectyfi Radar — V3 Ultimate Honest Baseline

## Mission Reference: VEC-52

This notebook is the third and most advanced iteration of the Vectyfi Radar baseline model.
It directly addresses every weakness discovered in V1 and V2 while maintaining an absolute
**zero target leakage** guarantee.

---

## Table of Contents

1. [Context & Motivation](#1-context--motivation)
2. [Dataset Overview](#2-dataset-overview)
3. [Cell-by-Cell Walkthrough](#3-cell-by-cell-walkthrough)
4. [The Leakage Exorcism (Deep Dive)](#4-the-leakage-exorcism-deep-dive)
5. [V3 Feature Engineering Blueprint](#5-v3-feature-engineering-blueprint)
6. [Class Imbalance Calibration](#6-class-imbalance-calibration)
7. [Evaluation Metrics Explained](#7-evaluation-metrics-explained)
8. [V2 vs V3 Comparison](#8-v2-vs-v3-comparison)
9. [What Comes Next (V4 Roadmap)](#9-what-comes-next-v4-roadmap)

---

## 1. Context & Motivation

EU public procurement generates over 9.5 million records per year. The raw data is published as
XML/CSV bulk exports by the EU Publications Office (TED — Tenders Electronic Daily).

The original V1 model achieved a suspiciously high **0.963 ROC-AUC** — but this was caused by
**target leakage**: columns like `NUMBER_OFFERS`, `AWARD_VALUE_EURO`, and `WIN_NAME` were
leaking future outcome information into the feature matrix. The model was essentially "cheating"
by seeing the answer before making its prediction.

V2 fixed the leakage by surgically removing all 21 CAN (Contract Award Notice) columns, but
only used 4 basic features (`CPV`, `ISO_COUNTRY_CODE`, `B_FRA_AGREEMENT`, `TYPE_OF_CONTRACT`),
achieving a modest **0.6793 ROC-AUC** with severe class imbalance (74% Awarded vs 26% Not Awarded).
The model almost never predicted "Not Awarded" correctly (Recall ≈ 2%).

**V3 attacks this from three angles:**
- Dramatically expand the leak-free feature space (15+ features vs 4)
- Inject temporal signals (`month`, `year`, `quarter`) that capture seasonal procurement cycles
- Algorithmically calibrate XGBoost with `scale_pos_weight` to force the model to take the minority
  class seriously

---

## 2. Dataset Overview

| Property | Value |
|---|---|
| **Source File** | `export_CFC_2018_2023.csv` |
| **Raw Rows** | ~7.7 million (lot-level records) |
| **After Deduplication** | ~2.1 million unique `ID_NOTICE_CN` |
| **Training Sample** | 500,000 randomly sampled notices |
| **Total Columns in CSV** | 64 |
| **Columns Loaded by V3** | 21 (carefully selected) |
| **Target Variable** | `y`: 1 = Awarded, 0 = Not Awarded |

The target is derived from CAN (Contract Award Notice) columns (`WIN_NAME` and `AWARD_VALUE_EURO`).
If either indicates a winner was selected, `y = 1`. These columns are then **permanently dropped**
before any model training occurs.

---

## 3. Cell-by-Cell Walkthrough

### Cell 1: Imports & Environment

```python
import pandas as pd, numpy as np, matplotlib.pyplot as plt, xgboost as xgb
from sklearn.metrics import precision_recall_curve, average_precision_score, roc_curve
```

**What it does:**
- Loads all required libraries (Pandas, NumPy, XGBoost, Scikit-Learn).
- Sets `matplotlib` to the `dark_background` theme.
- Defines a color palette (`TEAL`, `CORAL`, `GOLD`, `SLATE`) used across all visualizations.
- Suppresses warnings to keep the notebook output clean.

**Why:**
Consistent visual identity across plots. The dark theme with teal/emerald accents is the Vectyfi
Radar design language.

---

### Cell 2: Data Loading & Strict Deduplication

```python
USECOLS = ['ID_NOTICE_CN', 'CPV', 'ISO_COUNTRY_CODE', 'B_FRA_AGREEMENT', ...]
df_raw = pd.read_csv(file_path, usecols=available_cols, low_memory=False)
df_dedup = df_raw.drop_duplicates(subset=['ID_NOTICE_CN'], keep='first')
```

**What it does:**
1. Defines the exact 21 columns to load (out of 64 total) — this prevents Out-of-Memory crashes
   on machines with limited RAM.
2. Reads the CSV with `usecols` optimization (only the columns we need are loaded into memory).
3. Safely checks which columns actually exist in the file (graceful fallback for schema changes).
4. **Deduplicates** on `ID_NOTICE_CN` — the unique Contract Notice identifier. The raw CSV contains
   lot-level rows (one notice can have 50+ lots), but for award prediction we need one row per notice.
5. Samples 500,000 rows randomly (`random_state=42` for reproducibility) if the deduplicated
   dataset exceeds 500k.

**Why the deduplication matters:**
Without it, a single notice with 100 lots would be represented 100 times, massively inflating the
training set with near-identical rows. This causes the model to overfit on high-lot-count notices
and produces misleading accuracy metrics.

---

### Cell 3: Target Creation & The Leakage Exorcism

```python
df['y'] = ((df['WIN_NAME'].notna()) | (df['AWARD_VALUE_EURO'] > 0)).astype(int)
```

**What it does:**
1. **Creates the binary target `y`:** A notice is "Awarded" (`y=1`) if either:
   - `WIN_NAME` is not null (a winner was identified), OR
   - `AWARD_VALUE_EURO > 0` (a contract value was recorded)
2. **Computes the class imbalance ratio:** `scale_pos_weight = count(Not Awarded) / count(Awarded)`.
   This exact ratio is stored for use in Cell 5.
3. **Executes the Leakage Exorcism:** Permanently drops all 23 CAN/result columns.

**The 23 Dropped Columns:**

| Column | Why It Leaks |
|---|---|
| `WIN_NAME` | The winner's name — *this IS the answer* |
| `WIN_NATIONALID` | Winner's tax ID |
| `WIN_ADDRESS`, `WIN_TOWN`, `WIN_POSTAL_CODE`, `WIN_COUNTRY_CODE` | Winner's location |
| `B_AWARDED_TO_A_GROUP` | Whether it was awarded to a consortium |
| `B_CONTRACTOR_SME` | Whether the winner is an SME |
| `NUMBER_OFFERS` | Total bids received — only known after deadline |
| `NUMBER_TENDERS_SME` | SME bid count — only known after deadline |
| `NUMBER_TENDERS_OTHER_EU` | Cross-border bids |
| `NUMBER_TENDERS_NON_EU` | Non-EU bids |
| `NUMBER_OFFERS_ELECTR` | Electronic bids |
| `AWARD_VALUE_EURO` | Final contract value |
| `AWARD_VALUE_EURO_FIN_1` | Financial correction value |
| `AWARD_EST_VALUE_EURO` | Post-award estimated value |
| `DT_AWARD` | Date of award decision |
| `ID_AWARD` | Award identifier |
| `B_SUBCONTRACTED` | Whether subcontracting occurred |
| `INFO_ON_NON_AWARD` | Reason for non-award |
| `CONTRACT_NUMBER` | Contract registration number |
| `FUTURE_CAN_ID` | Link to the CAN record |
| `FUTURE_CAN_ID_ESTIMATED` | Estimated CAN link |

**Critical principle:** If a column's value is only knowable *after* the award decision, it cannot
be used as an input feature. Using it would be like predicting whether it will rain tomorrow by
checking tomorrow's weather report today.

---

### Cell 4: Advanced Feature Engineering (V3 NEW)

This is the heart of V3. We extract **15+ features** from the remaining leak-free columns.

#### 4.1 Temporal Features (from `DT_DISPATCH`)

```python
dt = pd.to_datetime(X_raw['DT_DISPATCH'], errors='coerce')
X['dispatch_month']   = dt.dt.month
X['dispatch_year']    = dt.dt.year
X['dispatch_quarter'] = dt.dt.quarter
```

**Why:** EU procurement follows strong seasonal patterns:
- Q4 (October–December) sees a "budget rush" as agencies spend remaining annual budgets.
- Summer months (July–August) have lower activity and different competitive dynamics.
- Year-over-year trends capture regulatory changes (e.g., new eForms in 2024).

#### 4.2 DURATION (Numeric with Median Imputation)

```python
X['duration'] = pd.to_numeric(X_raw['DURATION'], errors='coerce')
X['duration'] = X['duration'].fillna(median_dur)
```

**Why:** Contract duration is a powerful signal. Short-duration contracts (< 6 months) behave
differently from multi-year frameworks. Missing values are imputed with the median to avoid
introducing bias (mean would be skewed by outliers).

#### 4.3 VALUE_EURO (Log-Transformed Estimated Value)

```python
X['value_euro_log'] = np.log1p(pd.to_numeric(X_raw['VALUE_EURO'], errors='coerce').fillna(0))
```

**Important distinction:** `VALUE_EURO` is the **pre-award estimated value** published in the
Contract Notice (CN). This is NOT the final award value (`AWARD_VALUE_EURO`, which was dropped).
The estimated value is public knowledge before the deadline and is leak-free.

The `log1p` transformation compresses the extreme range (values span from €1 to €10 billion)
into a more XGBoost-friendly distribution.

#### 4.4 Binary Flags

Seven boolean columns are mapped from `Y`/`N` strings to `1`/`0` integers:

| Feature | Meaning |
|---|---|
| `is_framework` | Is this a framework agreement (multi-year umbrella contract)? |
| `is_eu_funded` | Is this project co-funded by EU structural funds? |
| `is_gpa` | Is this covered by the WTO Government Procurement Agreement? |
| `is_dps` | Is this a Dynamic Purchasing System? |
| `is_e_auction` | Will an electronic reverse auction be used? |
| `is_renewable` | Can the contract be renewed/extended? |
| `has_options` | Does the contract include optional extensions? |

#### 4.5 Low-Frequency Category Grouping

For high-cardinality categorical columns (`CPV`, `ISO_COUNTRY_CODE`, `MAIN_ACTIVITY`), categories
appearing in fewer than 1% of rows are merged into an `Other` bucket.

```python
cpv_counts = cpv_raw.value_counts(normalize=True)
rare_cpvs = cpv_counts[cpv_counts < 0.01].index
cpv_raw = cpv_raw.replace(rare_cpvs, 'Other')
```

**Why:** Without this, One-Hot Encoding would create hundreds of near-zero columns (e.g., CPV code
`03` — agricultural products — might appear in only 200 out of 500,000 rows). These sparse columns:
- Add noise without predictive value
- Increase memory consumption
- Slow down training
- Risk overfitting to rare categories

#### 4.6 One-Hot Encoding

After grouping, the remaining categorical columns (`cpv_group`, `country`, `contract_type`,
`cae_type`, `top_type`, `main_activity`) are converted to binary dummy variables using
`pd.get_dummies(drop_first=True)`.

---

### Cell 5: Algorithmic Calibration — The Imbalance Fix

```python
scale_weight = n_neg / n_pos
model = xgb.XGBClassifier(scale_pos_weight=scale_weight, ...)
```

**The Problem:** In V2, the dataset was ~74% Awarded / ~26% Not Awarded. XGBoost defaulted to
predicting "Awarded" almost every time (achieving 74% accuracy by doing nothing useful).
The Recall for the "Not Awarded" class was a catastrophic 2%.

**The Fix:** `scale_pos_weight` tells XGBoost to multiply the loss function for the minority class
by this ratio. If the ratio is 0.35 (meaning negatives are 35% of positives), XGBoost will
treat each minority sample as ~2.85x more important.

**Additional V3 model improvements:**
- `n_estimators=200` (up from 100): More boosting rounds for a richer model.
- `max_depth=6` (up from 5): Slightly deeper trees to capture feature interactions.
- `eval_metric='aucpr'`: Monitors the Precision-Recall AUC during training (not just log-loss).
- `eval_set=[(X_test, y_test)]` with `verbose=50`: Prints validation performance every 50 rounds
  so you can visually confirm the model is improving and not overfitting.

---

### Cell 6: Advanced Evaluation & Metrics

```python
roc_auc = roc_auc_score(y_test, y_prob)
pr_auc  = average_precision_score(y_test, y_prob)
```

**Outputs:**
1. **ROC-AUC** — The classic metric. Measures how well the model separates the two classes
   across all probability thresholds. Range: 0.5 (random) to 1.0 (perfect).
2. **PR-AUC (Precision-Recall AUC)** — The *true* metric for imbalanced datasets. Unlike ROC-AUC,
   PR-AUC is not inflated by the large number of true negatives. It directly measures how well
   the model identifies positives without flooding the results with false positives.
3. **Confusion Matrix** — Formatted as a 2x2 grid showing True Negatives, False Positives,
   False Negatives, and True Positives.
4. **Classification Report** — Per-class Precision, Recall, and F1-Score.

---

### Cell 7: Visualization Dashboard

Three publication-quality plots on a single dark-themed canvas:

| Plot | What It Shows |
|---|---|
| **ROC Curve** (Teal) | The tradeoff between True Positive Rate and False Positive Rate. The area under this curve = ROC-AUC. The dashed diagonal = random guessing. |
| **Precision-Recall Curve** (Coral) | The tradeoff between Precision (of positive predictions) and Recall (coverage of actual positives). The horizontal dashed line = baseline PR (proportion of positives in the dataset). |
| **Top 15 Feature Importances** (Teal bars) | Which features XGBoost relies on most heavily. This reveals whether the model learned meaningful patterns or is over-relying on a single feature. |

---

## 4. The Leakage Exorcism (Deep Dive)

Target leakage is the single most dangerous mistake in applied machine learning. It occurs when
information that would not be available at prediction time leaks into the training data.

In the procurement context:
- **At prediction time:** A Contract Notice (CN) has just been published. We want to predict
  whether it will result in an award.
- **Available information:** The CN itself (CPV code, country, contract type, estimated value,
  deadline, procedure type, etc.)
- **NOT available:** Anything from the future CAN (Contract Award Notice) — winner name,
  final value, number of bids received, etc.

Our exorcism drops **every single column** that originates from the CAN record. This is verified
by cross-referencing the official TED data dictionary.

---

## 5. V3 Feature Engineering Blueprint

```
Raw CSV (64 columns)
    │
    ├── Load 21 selected columns (usecols optimization)
    │
    ├── Drop 23 CAN/leak columns
    │
    ├── Temporal: DT_DISPATCH → month, year, quarter
    │
    ├── Numeric: DURATION → median imputed float
    │         VALUE_EURO → log1p transformed
    │         LOTS_NUMBER → integer
    │
    ├── Binary: 7 Y/N flags → 0/1
    │
    ├── Categorical: CPV → 2-digit group → low-freq → Other
    │             Country → low-freq → Other
    │             Contract Type, CAE Type, TOP Type, Main Activity
    │
    └── One-Hot Encode remaining categoricals
         │
         └── Final X_encoded matrix (~100-150 columns)
```

---

## 6. Class Imbalance Calibration

| Class | Count | Percentage |
|---|---|---|
| Awarded (y=1) | ~370,000 | ~74% |
| Not Awarded (y=0) | ~130,000 | ~26% |

Without calibration, XGBoost learns that predicting "Awarded" every time yields 74% accuracy.
The minority class is essentially invisible.

`scale_pos_weight = count(y=0) / count(y=1) ≈ 0.35`

This forces XGBoost to weight minority-class errors 2.85x more heavily, dramatically improving
Recall on the "Not Awarded" class — which is the commercially valuable prediction (identifying
tenders that will *fail* helps clients avoid wasting bid preparation resources).

---

## 7. Evaluation Metrics Explained

| Metric | What It Measures | Good Value |
|---|---|---|
| **ROC-AUC** | Overall discrimination ability | > 0.70 (honest baseline) |
| **PR-AUC** | Precision-Recall balance on imbalanced data | > 0.80 (given 74% positive rate) |
| **Precision** | Of all predicted positives, how many are correct? | Higher = fewer false alarms |
| **Recall** | Of all actual positives, how many did we find? | Higher = fewer missed opportunities |
| **F1-Score** | Harmonic mean of Precision and Recall | Balanced tradeoff |

---

## 8. V2 vs V3 Comparison

| Dimension | V2 Honest Baseline | V3 Ultimate Baseline |
|---|---|---|
| **Input Features** | 4 (CPV, Country, FRA, Contract) | 15+ (structural + temporal + numeric) |
| **Temporal Signals** | None | `dispatch_month`, `dispatch_year`, `dispatch_quarter` |
| **DURATION** | Ignored | Numeric with median imputation |
| **VALUE_EURO** | Ignored | log1p transformed (leak-free estimated value) |
| **Binary Flags** | 1 (`B_FRA_AGREEMENT`) | 7 (EU funds, GPA, DPS, e-auction, renewals, options) |
| **Category Grouping** | None (all categories kept) | <1% frequency → `Other` |
| **Class Imbalance** | Not addressed (`scale_pos_weight=1`) | Auto-calculated `scale_pos_weight` |
| **XGBoost Trees** | 100 estimators, depth 5 | 200 estimators, depth 6 |
| **Eval Metric** | ROC-AUC only | ROC-AUC + PR-AUC + PR Curve |
| **Visualization** | 1 chart (feature importance) | 3-panel dashboard (ROC, PR, Features) |
| **ROC-AUC** | 0.6793 | *(expected: significant improvement)* |
| **Target Leakage** | Zero | Zero |

---

## 9. What Comes Next (V4 Roadmap)

V3 represents the ceiling of what pure tabular, manually engineered features can achieve on this
dataset. The next iterations will break through this ceiling:

1. **V4: Unsupervised Micro-Market Clustering**
   Run K-Means / HDBSCAN on `all-MiniLM-L6-v2` embeddings of the raw tender descriptions.
   Assign a `cluster_id` to every row. XGBoost gets a new categorical feature representing
   the tender's "market niche" — learned entirely from the text, not from human rules.

2. **V5: The LLM Sniper Feature Extraction**
   Use Ollama + Gemma 8B (running locally on Apple Silicon M4 Pro) to extract 6 deterministic
   commercial features from the unstructured XML text:
   - `is_framework_agreement` (Boolean)
   - `subcontracting_allowed` (Boolean)
   - `contract_duration_months` (Integer)
   - `tech_domain` (Categorical)
   - `extracted_budget_eur` (Float)
   - `requires_security_clearance` (Boolean)

3. **V6: Optuna Hyperparameter Optimization**
   Full Bayesian hyperparameter search over XGBoost's parameter space using Optuna.

4. **V7: Production Pipeline Serialization**
   Export the final model + preprocessing pipeline as a production-ready artifact and connect
   it to the FastAPI `/predict` endpoint.
