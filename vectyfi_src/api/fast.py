from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import numpy as np
import pandas as pd
import random
from pathlib import Path
import io
import shap
from sklearn.pipeline import Pipeline

from vectyfi_src.interface.main import preprocess
from vectyfi_src.ml.explainer import build_explainer, explain_instance

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = Path(__file__).parent.parent.parent / "ml" / "model.pkl"

try:
    with open(MODEL_PATH, "rb") as f:
        loaded = pickle.load(f)
    model = loaded['model'] if isinstance(loaded, dict) else loaded
    print(f"✅ Model loaded from {MODEL_PATH}")
except FileNotFoundError:
    raise RuntimeError(f"model_test.pkl introuvable à {MODEL_PATH}")

explainer = shap.Explainer(model)

DUMMY_VALUES = {
    "B_MULTIPLE_CAE": ["Y", "N"], "B_EU_FUNDS": ["Y", "N"],
    "B_GPA": ["Y", "N"], "B_FRA_AGREEMENT": ["Y", "N"], "B_ACCELERATED": ["Y", "N"],
    "LOTS_NUMBER": [1.0, 3.0, 5.0, 10.0],
    "YEAR": [2018, 2019, 2020, 2021, 2022, 2023, 2024],
    "CRIT_PRICE_WEIGHT": [0.0, 30.0, 50.0, 60.0, 80.0],
    "CRIT_CODE": [0.0, 1.0],
    "TOP_TYPE": ["OPE", "AWP", "NIC", "RES", "NOP", "NOC", "NIP", "COD", "INP"],
    "ISO_COUNTRY_CODE": ["PL", "ES", "FR", "BG", "RO", "SE", "IT", "DK", "DE", "NO"],
    "TYPE_OF_CONTRACT": ["U", "S", "W"],
    "CAE_TYPE": ["6", "3", "1", "4", "8", "R", "N", "5", "Z", "5A"],
    "MAIN_ACTIVITY": [
        "Health", "Defence", "Railway services", "Other",
        "General public\\services", "Education", "Environment",
        "Urban railway, tramway, trolleybus or bus services",
        "Housing and community amenities", "Recreation, culture and religion"
    ],
}

class TenderInput(BaseModel):
    # Binary
    B_MULTIPLE_CAE:  str
    B_EU_FUNDS:      str
    B_GPA:           str
    B_FRA_AGREEMENT: str
    B_ACCELERATED:   str

    # Numerical
    LOTS_NUMBER:       float
    YEAR:              int
    CRIT_PRICE_WEIGHT: float
    CRIT_CODE:         float

    # Categorical
    TOP_TYPE:          str
    ISO_COUNTRY_CODE:  str
    TYPE_OF_CONTRACT:  str
    CAE_TYPE:          str
    MAIN_ACTIVITY:     str

def prepare_input(data: dict) -> pd.DataFrame:
    df = pd.DataFrame([data])
    df['INFO_ON_NON_AWARD'] = 'awarded'
    X, _ = preprocess(df)

    # Drop INFO_ON_NON_AWARD — model was trained without this column
    X = X.drop(columns=["INFO_ON_NON_AWARD"], errors="ignore")

    # Reorder columns to match exactly what the model expects
    X = X[model.feature_names_in_]

    # Cast string columns to pandas Categorical — model was trained with this dtype
    for col in X.select_dtypes(include=["object"]).columns:
        X[col] = X[col].astype("category")  # keep as category, NOT .cat.codes

    return X

def force_plot_html(X_preprocessed: pd.DataFrame) -> str:
    X_encoded = X_preprocessed.copy()

    # SHAP/XGBoost DMatrix needs integer codes, not category dtype
    for col in X_encoded.select_dtypes(include=["category"]).columns:
        X_encoded[col] = X_encoded[col].cat.codes

    sv = explainer(X_encoded)[0]

    force = shap.plots.force(
        base_value=sv.base_values,
        shap_values=sv.values,
        features=X_preprocessed.iloc[0].tolist(),  # original values for display
        feature_names=X_preprocessed.columns.tolist(),
        matplotlib=False,
    )
    buf = io.StringIO()
    shap.save_html(buf, force)
    return buf.getvalue()

@app.get("/")
def root():
    return {"message": "Vectyfi, Tender Prediction API!"}

#input from user
@app.post("/predict")
def predict(data: TenderInput):
    X = prepare_input(data.model_dump())
    prediction = model.predict(X)
    proba = model.predict_proba(X)[0]
    return {
        "input": data.model_dump(),
        "accepted": bool(prediction[0]),
        "confidence": round(float(max(proba)), 2),
        "force_plot_html": force_plot_html(X)
    }

@app.post("/explain")
def explain(data: TenderInput):
    X = prepare_input(data.model_dump())
    return {"force_plot_html": force_plot_html(X)}

#input random for testing
@app.get("/dummy/predict")
def dummy_predict():
    dummy = {key: random.choice(values) for key, values in DUMMY_VALUES.items()}
    X = prepare_input(dummy)
    prediction = model.predict(X)
    proba = model.predict_proba(X)[0]
    return {
        "input": dummy,
        "accepted": bool(prediction[0]),
        "confidence": round(float(max(proba)), 2)
    }
