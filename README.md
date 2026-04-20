# 🦅 Vectyfi Radar: Predictive Procurement Intelligence
Vectyfi Radar is a data-driven B2B SaaS engine designed to solve the procurement haystack problem. EU public procurement generates over 3.6GB of unstructured data across 9.5M+ XML/CSV records. This repository houses Project Zero: the core Machine Learning engine built to parse, clustering, and evaluate these records.

By combining rigorous statistical inference, unsupervised micro-market clustering, and local LLM semantic extraction, this engine predicts contract award probabilities to identify high-value "Golden Leads" for mid-sized enterprises.

## 🛠️ The Tech Stack
We decoupled our architecture into two distinct layers: a highly responsive web shell and a mathematically rigorous, locally executed Machine Learning backend.

**Core Architecture & Data Engineering**

* **Backend:** FastAPI (Python) for robust, high-performance API endpoints.
* **Frontend:** React with a strict "Bio-Digital" design system.
* **Database:** PostgreSQL, utilizing the pgvector extension for high-dimensional vector storage.
* **Data Manipulation:** Pandas and NumPy for handling massive procurement datasets.

**Machine Learning & Statistics (The Engine)**

* **Statistical Foundation:** Rigorous statistical inference (hypothesis testing, confidence intervals) to mathematically validate feature distributions before modeling.
* **Unsupervised Learning:** Scikit-Learn (K-Means / HDBSCAN) for dynamic "Micro-Market" clustering.
* **Predictive Modeling:** XGBoost for complex, non-linear classification, optimized via Optuna for hyperparameter tuning.

**Generative AI & NLP (The "LLM Sniper" Pipeline)**

* **Semantic Embeddings:** HuggingFace all-MiniLM-L6-v2 for fast, local vectorization of unstructured XML texts into 384-dimensional space.
* **Feature Extraction:** Ollama running Google Gemma 8B (in q8_0) natively on Apple Silicon to extract deterministic commercial features via strict JSON prompting.

## 📁 Project Structure
* `ml_logic/`: Core Python package containing the ML pipeline (data cleaning, preprocessing, modeling).
* `notebooks/`: Jupyter notebooks for EDA, statistical inference, and sandbox experiments.
* `scripts/`: Executable Python scripts for data ingestion and automated LLM extraction.
* `tests/`: Unit tests for the core logic.
