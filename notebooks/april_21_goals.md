# 🦅 Vectyfi Radar - Daily Sync & Sprint Plan

## 1. Model Shootout & Baseline Selection
Let's review the models built so far. We will compare the results and officially lock in our baseline before moving forward.

| Team Member | Model Version / Approach | ROC-AUC | Notes / Key Takeaways |
| :--- | :--- | :--- | :--- |
| [Name] | *e.g., V3 (XGBoost, Imbalance Fix)* | *0.783* | *Good handling of False Positives.* |
| [Name] | | | |
| [Name] | | | |

**🏆 Official Baseline Selected:** [To be filled during meeting]

---

## 🛠 Task Board (Parallel Workstreams)

### [VEC-101] XML Feature Sniper (LLM Extraction)
* **Assignee:** [Name]
* **Goal:** Build a local Ollama/Gemma-8B pipeline.
* **Agile Scope:** Do not wait for the perfect V3 filtering. Take a random sample of 10,000 raw XML descriptions from the DB and prove that the LLM can reliably extract 5 JSON features (e.g., subcontracting, strict deadlines).
* **Expected Outcome:** A working script and a test CSV (`llm_test_features.csv`).
* **Real Outcome:** [To be filled]

---

### [VEC-102] Semantic Micro-Market Clustering
* **Assignee:** [Name]
* **Goal:** Build the clustering algorithm (HDBSCAN/K-Means) on the MiniLM embeddings.
* **Agile Scope:** You do not need the V3 model. Pull 50,000 vectors directly from the Vector-DB, run the algorithm, and determine if the resulting clusters make economic sense.
* **Expected Outcome:** A Jupyter Notebook showing the Silhouette scores and assigning a `semantic_cluster_id` to each tender in the sample.
* **Real Outcome:** [To be filled]

---

### [VEC-103] Hybrid Architecture Stubbing (Mock Integration)
* **Assignee:** [Name]
* **Goal:** Prepare the V3 XGBoost pipeline to ingest external AI features.
* **Agile Scope:** **Do not wait for Clemens or Basile!** Create "mock data" (fake columns like `mock_cluster_id` or `mock_llm_feature` with random values) and build the code that safely merges these features into the V3 model via `ID_NOTICE_CN` without crashing or causing Target Leakage.
* **Expected Outcome:** A tested code architecture (a `merge_and_train()` function) where we only need to plug in the real data from Clemens and Basile tomorrow.
* **Real Outcome:** [To be filled]

---

### [VEC-104] XML Bulk Parsing Infrastructure
* **Assignee:** [Name]
* **Goal:** Build a robust, memory-efficient XML parser for the 2024-2026 bulks.
* **Agile Scope:** Focus purely on the Data Engineering pipeline. The script must stream massive XML files (without loading them entirely into RAM), clean the texts, and prepare them for database insertion.
* **Expected Outcome:** A Python script (`xml_stream_parser.py`) that runs without Out-of-Memory errors.
* **Real Outcome:** [To be filled]




# 🎯 [VEC-XXX] Task Title

**Assignee:** @[Name]
**Phase:** Phase 2 - XML & Vector Evolution

## 📝 Context & Goal
*Why are we doing this and what is the specific objective?*
> 

## 🛠 Proposed Plan & Methodology
*How will you approach this? (Algorithms, libraries, data sources)*
* Step 1: 
* Step 2: 
* Step 3: 

## ✅ Definition of Done (Acceptance Criteria)
*What must be true for this task to be considered complete?*
* [ ] 
* [ ] 
* [ ] Code pushed to the `edu-dev` branch without target leakage.

## 🔮 Expected Outcome
*What do you expect the result or impact on the V3 baseline to be?*
> 

---
*(Stop here during planning. Fill the section below when the task is finished.)*
---

## 📊 Actual Outcome & Learnings
*What actually happened? What were the final metrics, roadblocks, or surprising findings?*
>