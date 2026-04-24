# 🦅 Vectyfi Radar: Predictive Procurement Intelligence
Vectyfi Radar is a data-driven B2B SaaS engine designed to solve the procurement haystack problem. EU public procurement generates over 3.6GB of unstructured data across 9.5M+ XML/CSV records. This repository houses Project Zero: the core Machine Learning engine built to parse, clustering, and evaluate these records.

By combining rigorous statistical inference, unsupervised micro-market clustering, and local LLM semantic extraction, this engine predicts contract award probabilities to identify high-value "Golden Leads" for mid-sized/small and medium enterprises enterprises (SME).

## 🛠️ The Tech Stack
We decoupled our architecture into two distinct layers: a highly responsive web shell and a mathematically rigorous, locally executed Machine Learning backend.

**Core Architecture & Data Engineering**

* **Backend:** FastAPI (Python 3.14.3) for robust, high-performance API endpoints.
* **Frontend:** React with a strict "Bio-Digital" design system.
* **Database:** PostgreSQL, utilizing the pgvector extension for high-dimensional vector storage.
* **Data Manipulation:** Pandas and NumPy for handling massive procurement datasets.

**Machine Learning & Statistics (The Engine)**

* **Statistical Foundation:** Rigorous statistical inference (hypothesis testing, confidence intervals) to mathematically validate feature distributions before modeling.
* **Unsupervised Learning:** Scikit-Learn (K-Means / HDBSCAN) for dynamic "Micro-Market" clustering.
* **Predictive Modeling:** XGBoost for complex, non-linear classification with a leak-free sklearn Pipeline (ColumnTransformer + XGBClassifier).

**Generative AI & NLP (The "LLM Sniper" Pipeline)**

* **Semantic Embeddings:** HuggingFace all-MiniLM-L6-v2 for fast, local vectorization of unstructured XML texts into 384-dimensional space.
* **Feature Extraction:** Ollama running Google Gemma 8B (in q8_0) natively on Apple Silicon to extract deterministic commercial features via strict JSON prompting.

## 📁 Project Structure
```
radar.vectyfi/
├── vectyfi_src/
│   ├── interface/
│   │   └── main.py           # CLI entry point (clean → train → evaluate → predict)
│   ├── ml/
│   │   ├── data_cleaning.py  # Raw TED CSV → balanced, leak-free dataset
│   │   └── preprocessing.py  # Feature encoding + sklearn Pipeline builder
│   ├── api/                   # FastAPI server (serves model_test.pkl)
│   └── frontend/              # React frontend
├── notebooks/                 # Jupyter notebooks for EDA & experiments
├── raw_data/                  # Input CSVs and cleaned datasets
├── scripts/                   # Data ingestion & LLM extraction scripts
└── tests/                     # Unit tests
```

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the ML pipeline

```bash
# Full pipeline: clean raw data → train → evaluate → predict
python -m vectyfi_src.interface.main

# Skip cleaning — reuse the existing balanced CSV (much faster)
python -m vectyfi_src.interface.main --skip-clean

# Train only, no .pkl saved to disk (quick experiment)
python -m vectyfi_src.interface.main --skip-clean --no-save
```

| Flag | Effect |
|---|---|
| `--skip-clean` | Skip data cleaning; loads `raw_data/f_balanced_500k.csv` directly |
| `--no-save` | Do NOT export the trained pipeline to `model_test.pkl` |

### 3. Expected output
```
Model AUC:      0.7582
Model Accuracy: 68.1%
```
AUC of ~0.76 on 14 pre-award procedural features with a 50/50 balanced dataset (500k rows). No data leakage.

---

## 🔌 Using `model_test.pkl` in the API

The training pipeline exports a **full sklearn Pipeline** (not just the XGBoost model). This means the `.pkl` file contains:
1. **ColumnTransformer** — handles all feature encoding (OneHot, TargetEncoder, passthrough)
2. **XGBClassifier** — the trained model

You can load it on any server and call `.predict()` directly on raw feature DataFrames — no manual re-encoding needed.

### Minimal FastAPI example

```python
# vectyfi_src/api/main.py
import pickle
import pandas as pd
from pathlib import Path
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Vectyfi Radar API")

# Load the full pipeline once at startup
MODEL_PATH = Path("model_test.pkl")
with open(MODEL_PATH, "rb") as f:
    pipeline = pickle.load(f)


class ProcurementInput(BaseModel):
    """Input schema matching the 14 pre-award TED features."""
    B_MULTIPLE_CAE: str    # "Y" or "N"
    B_EU_FUNDS: str        # "Y" or "N"
    B_FRA_AGREEMENT: str   # "Y" or "N"
    B_GPA: str             # "Y" or "N"
    B_ACCELERATED: str     # "Y" or "N"
    TOP_TYPE: str           # e.g. "OPEN"
    ISO_COUNTRY_CODE: str   # e.g. "FR"
    TYPE_OF_CONTRACT: str   # e.g. "SERVICES"
    CAE_TYPE: str           # e.g. "BODY_PUBLIC"
    MAIN_ACTIVITY: str      # e.g. "GENERAL_PUBLIC_SERVICES"
    CRIT_CODE: str          # "L" or "M"
    CRIT_PRICE_WEIGHT: float
    LOTS_NUMBER: int
    YEAR: int


@app.post("/predict")
def predict(data: ProcurementInput):
    """
    Predict whether a procurement contract will be awarded or not.

    Returns:
        prediction: 0 = awarded, 1 = not awarded
        probability: probability of non-award (float 0-1)
    """
    # Convert input to DataFrame (pipeline expects DataFrame input)
    df = pd.DataFrame([data.model_dump()])

    prediction = int(pipeline.predict(df)[0])
    proba = float(pipeline.predict_proba(df)[0, 1])

    return {
        "prediction": prediction,
        "label": "not_awarded" if prediction == 1 else "awarded",
        "probability_not_awarded": round(proba, 4),
    }


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": pipeline is not None}
```

### Deploy and call
```bash
# Start the API server
uvicorn vectyfi_src.api.main:app --host 0.0.0.0 --port 8000

# Test with curl
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "B_MULTIPLE_CAE": "N",
    "B_EU_FUNDS": "N",
    "B_FRA_AGREEMENT": "N",
    "B_GPA": "N",
    "B_ACCELERATED": "N",
    "TOP_TYPE": "OPEN",
    "ISO_COUNTRY_CODE": "FR",
    "TYPE_OF_CONTRACT": "SERVICES",
    "CAE_TYPE": "BODY_PUBLIC",
    "MAIN_ACTIVITY": "GENERAL_PUBLIC_SERVICES",
    "CRIT_CODE": "L",
    "CRIT_PRICE_WEIGHT": 60.0,
    "LOTS_NUMBER": 1,
    "YEAR": 2023
  }'
```

### Key points for deployment
- The `.pkl` contains the **full pipeline** — just `pickle.load()` and call `.predict()`.
- The API input expects **raw TED values** (e.g. `"Y"`/`"N"` for binary flags). The pipeline handles encoding internally.
- Copy `model_test.pkl` to your server alongside the API code. No other data files needed.
- For production, consider using `joblib` instead of `pickle` for slightly faster serialization.
