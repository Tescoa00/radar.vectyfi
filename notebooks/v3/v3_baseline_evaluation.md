# V3 Honest Baseline — Expert Evaluation

## Verdict: Strong Baseline, With Caveats

The ROC-AUC of **0.9193** is genuinely excellent for a leakage-free baseline built on purely categorical/structural features. This model has real predictive signal. However, several issues suppress its operational value, and one metric is misleading.

---

## 1. The PR-AUC of 0.9793 Is Inflated

> [!WARNING]
> The PR-AUC is computed for the **majority class** (Awarded, class 1), not the minority class you actually care about predicting.

`average_precision_score(y_test, y_prob)` computes precision-recall with respect to the **positive label** (y=1 = Awarded). Since Awarded makes up ~81% of the data, the *random baseline* for this metric is already ~0.81. Your 0.9793 is above that, but not as spectacular as it first appears.

**What to do**: Compute PR-AUC for the minority class to get the real picture:
```python
# PR-AUC for the minority class (Not Awarded = 0)
pr_auc_minority = average_precision_score(1 - y_test, 1 - y_prob)
print(f"PR-AUC (Not Awarded): {pr_auc_minority:.4f}")
```
This will likely be significantly lower and more honest about predictive power on the class that matters.

---

## 2. Classification Report — The Real Story

| Class | Precision | Recall | F1 | Interpretation |
|---|---|---|---|---|
| Not Awarded (0) | **0.52** | 0.83 | 0.64 | ⚠️ Half of "risk" flags are false alarms |
| Awarded (1) | 0.95 | 0.82 | 0.88 | ✅ Solid majority-class performance |

The critical weakness is the **52% precision on the minority class**. In a production Radar system, this means:
- For every 100 contracts flagged as "will NOT be awarded," **~48 of them would actually get awarded**.
- The **83% recall** is decent — the model catches most actual non-awards.
- This precision/recall trade-off is a direct consequence of the `scale_pos_weight` setting, which aggressively pushes the model toward minority-class recall at the expense of precision.

> [!NOTE]
> For a baseline, this trade-off is acceptable. In production, you'd tune the decision threshold (currently 0.5) to find the sweet spot between recall and precision based on business needs.

---

## 3. `scale_pos_weight` Usage — Correct but Counterintuitive

The calculation `not_awarded / awarded = 0.2354` is **technically correct** for XGBoost's formula. It *downweights* the positive (Awarded) class, giving the gradient from negative (Not Awarded) instances relatively more influence.

However, the printed value in training says `scale_pos_weight=0.2275` while the calculated value is `0.2354`. This suggests the model cell was run with stale state from a previous notebook execution where the data pipeline was slightly different (maybe before the direct-award filter was applied, or with different dedup).

> [!IMPORTANT]
> **The training cell (Cell 4) has `execution_count: 15` while Cells 1-3 have `execution_count: 17, 18, 19`.** This confirms the model was trained on a **previous run's data**, not the current pipeline output. The displayed results are from `scale_pos_weight=0.2275`, not the currently calculated `0.2354`. You should **re-run Cell 4** after Cells 1-3 to ensure consistency.

---

## 4. Methodological Issues

### 4a. Test set used as `eval_set` during training
```python
xgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], ...)
```
While XGBoost does **not** use `eval_set` for gradient updates (no leakage), this is still poor practice. If you later add early stopping, the test set would directly influence when training stops, contaminating your evaluation. Use a separate validation split:

```python
X_train2, X_val, y_train2, y_val = train_test_split(
    X_train, y_train, test_size=0.15, stratify=y_train, random_state=42
)
xgb_model.fit(X_train2, y_train2, eval_set=[(X_val, y_val)], ...)
# Then evaluate on X_test/y_test (never seen by the model or eval)
```

### 4b. No out-of-time validation
The current train/test split is **random** (stratified). For 2018–2023 procurement data, this means the model sees future data during training and predicts past data during testing — the opposite of real-world deployment.

An **out-of-time split** (e.g., train on 2018-2022, test on 2023) would be far more scientifically rigorous and realistic.

### 4c. Feature importance uses `weight` (frequency), not `gain`
`importance_type='weight'` counts how often a feature appears in tree splits. This can be misleading — a noisy feature might be split on many times without contributing much predictive power. Use `gain` or SHAP values for a more meaningful ranking:

```python
xgb.plot_importance(xgb_model, max_num_features=15, importance_type='gain')
```

---

## 5. Feature Engineering Assessment

| Step | Verdict |
|---|---|
| Dedup by `ID_AWARD` | ✅ Correct — preserves multi-lot structure |
| Remove direct awards (NOC/NOP/AWP) | ✅ Sound domain logic |
| `INFO_ON_NON_AWARD` → target | ✅ Clean, no leakage |
| `CRIT_PRICE_WEIGHT` cleaning | ⚠️ NaN → 0 conflates "missing" with "0% weight" — add a missing flag |
| `AWARD_EST_VALUE_EURO` → log + missing flag | ✅ Good handling of structural missingness |
| CPV → 2-digit division | ✅ Reasonable dimensionality reduction |
| Low-frequency grouping (1% threshold) | ✅ Prevents high-cardinality one-hot explosion |
| Binary flags (Y/N → 1/0) | ✅ Clean |

---

## 6. Feature Importance — Domain Interpretation

The top features make strong domain sense:

1. **CRIT_PRICE_WEIGHT** (981) — How much procurement weight is on price. Higher price weight often correlates with simpler, more successful procurements.
2. **B_FRA_AGREEMENT** (607) — Framework agreements have fundamentally different award dynamics.
3. **CRIT_CODE_UNKNOWN / CRIT_CODE_M** — Award criteria type strongly predicts outcomes.
4. **TOP_TYPE_OPE** (332) — Open procedures vs. restricted/competitive dialogue.
5. **CPV_DIVISION_33** (326) — Medical equipment sector (high-value, complex procurement).
6. **Country codes (SI, PL, CZ)** — Country-specific procurement cultures and success rates differ significantly.

> [!TIP]
> The model is learning real procurement dynamics, not spurious correlations. This is a good sign for the baseline's validity.

---

## 7. Summary Scorecard

| Dimension | Score | Notes |
|---|---|---|
| Leakage prevention | ⭐⭐⭐⭐⭐ | Exemplary — no post-award features |
| ROC-AUC | ⭐⭐⭐⭐☆ | 0.92 is excellent for a structural-feature baseline |
| PR-AUC reporting | ⭐⭐☆☆☆ | Misleading — computed for majority class |
| Minority-class precision | ⭐⭐☆☆☆ | 52% — too many false alarms for production |
| Validation strategy | ⭐⭐☆☆☆ | Random split, not out-of-time |
| Reproducibility | ⭐⭐⭐☆☆ | Cell execution order mismatch detected |
| Feature engineering | ⭐⭐⭐⭐☆ | Clean, with minor CRIT_PRICE_WEIGHT NaN issue |
| Domain validity | ⭐⭐⭐⭐⭐ | Feature importances align with procurement domain knowledge |

---

## 8. Recommended Next Steps

1. **Re-run Cell 4** after Cells 1-3 to ensure consistent `scale_pos_weight`
2. **Add minority-class PR-AUC** to the evaluation output
3. **Implement out-of-time validation** (train ≤2022, test 2023)
4. **Add a separate validation split** for `eval_set` (don't use test data)
5. **Switch feature importance to `gain`** or add SHAP analysis
6. **Add a missing-indicator for `CRIT_PRICE_WEIGHT`** before filling NaN with 0
7. **Explore threshold tuning** — the default 0.5 threshold is rarely optimal for imbalanced data
