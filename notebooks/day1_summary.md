# Day 1 Summary: The Evolution to a Mathematically Honest Baseline

This document summarizes our Day 1 progress in the Data Science sprint, strictly documenting how we moved from a flawed prototype to a robust, scientifically valid V3 baseline.

---

## 1. The Starting Point: `vectyfi_radar_preview` (V1)
**The Issue:** Target Leakage
When we first ran the V1 prototype notebook, the model achieved a spectacular **0.963 ROC-AUC**. However, upon deep inspection, we discovered the dataset was saturated with **Target Leakage**.

Target leakage happens when the machine learning model is trained using information that would not be available in a real-world predictive scenario. In V1, the model was using features from the Contract Award Notice (CAN) — such as the `WIN_NAME` (the winner's name), `NUMBER_OFFERS` (how many companies bid), and the final `AWARD_VALUE_EURO`. In reality, when we are making a prediction, the tender has just been published (Contract Notice) and none of this future information exists yet.

---

## 2. The Correction: V2 Honest Baseline
**What we tried:** To fix V1, we built the V2 notebook and performed a "Leakage Exorcism". We surgically dropped the 21 columns that contained future Award data. We then trained an XGBoost model using only 4 basic, leak-free features: `CPV` (Industry), `ISO_COUNTRY_CODE` (Location), `B_FRA_AGREEMENT` (Is it a framework?), and `TYPE_OF_CONTRACT`.

**The Outcome:** 
* The ROC-AUC dropped to a mathematically truthful **0.6793**.
* **The New Problem:** We discovered a severe **Class Imbalance**. About 74% of the dataset results in an "Awarded" status, and only 26% is "Not Awarded". Because we didn't help the algorithm handle this imbalance, XGBoost simply guessed "Awarded" for almost everything. It achieved a 74% accuracy, but its ability to predict "Not Awarded" (the negative class) was a catastrophic **2% Recall**.

---

## 3. The Ultimate Fix: V3 Radar Vectyfi Baseline
The V3 notebook is our definitive, honest baseline. It solves the V2 imbalance and dramatically expands our feature engineering while guaranteeing zero target leakage.

### Which Columns are Used & Why?
We expanded the feature set from 4 to over 15 high-value columns:
* **`DT_DISPATCH` (Temporal):** We extracted `month`, `year`, and `quarter`. This allows XGBoost to learn EU seasonal procurement trends (e.g., Q4 massive budget flushes).
* **`DURATION` (Numeric):** Extremely predictive. Multi-year frameworks behave differently than 3-month spot buys. We forced this into a float and used median-imputation for missing values.
* **`VALUE_EURO` (Estimated Budget):** We log-transformed this array. *Crucial distinction:* This is the pre-award *estimated* value published alongside the tender. It is 100% leak-free, unlike the final award value!
* **Low-Frequency Groups:** We grouped `CPV` and `ISO_COUNTRY_CODE` values that appeared in less than 1% of the data into an `"Other"` bucket. This removed massive noise and stabilized the One-Hot Encoding.

### Solving Data Leakage & Class Imbalance
* **Leakage Exorcism:** Exact same mechanism as V2. The 21 CAN columns are permanently deleted before the `X` feature matrix is created.
* **Algorithmic Calibration:** To fix the 2% Recall failure, we calculated the exact imbalance ratio (`scale_pos_weight = count(Not Awarded) / count(Awarded)`). By injecting this `scale_pos_weight` into XGBoost, we violently penalize the model whenever it misses a "Not Awarded" class. It forces the model to take the 26% minority class seriously.

### Upgraded Evaluation
Because of the 74/26 imbalance, ROC-AUC is misleading. In V3, we introduced **Precision-Recall AUC (PR-AUC)** as our primary "True Metric," alongside standard Confusion Matrices and Feature Importance plots.

---

## Q&A: Day 1 Bootcamp Review

**Q: Why did we sample only 500k rows instead of using the full 2.1 million deduplicated rows?**
**A:** To ensure rapid iteration during EDA. 500,000 rows is mathematically more than enough to capture the variance of the market for a baseline model without causing Out-of-Memory (OOM) errors on local machines. Once the script is finalized, V4 will scale to the full dataset.

**Q: If we dropped the `VALUE_EURO` (Award) column to stop leakage, how did we create the `y` target?**
**A:** We use the CAN award columns *exclusively* to create the 1/0 `y` array. Once `y` is safely secured in memory, we destroy those CAN columns from the `X` matrix so the algorithm can never see them during training.

**Q: Why don't we use standard textual keywords instead of AI embeddings for the text?**
**A:** EU procurement is multilingual (German, Polish, French, etc.) and uses massive synonym overlap. Standard keyword searches fail. Tomorrow, in V4, we will introduce `all-MiniLM-L6-v2` embeddings, transforming text into 384-dimensional mathematical vectors so we can search by *semantic business meaning* rather than just strings.
