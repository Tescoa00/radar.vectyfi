from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import numpy as np
import pandas as pd
import random
from pathlib import Path

from vectyfi_src.interface.main import preprocess

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = Path(__file__).parent / "model_test.pkl"

try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    print(f"✅ Model loaded from {MODEL_PATH}")
except FileNotFoundError:
    raise RuntimeError(f"model_test.pkl introuvable à {MODEL_PATH}")

DUMMY_VALUES = {
    "B_MULTIPLE_CAE": ["Y", "N"], "B_EU_FUNDS": ["Y", "N"],
    "B_GPA": ["Y", "N"], "B_FRA_AGREEMENT": ["Y", "N"], "B_ACCELERATED": ["Y", "N"],
    "LOTS_NUMBER": [1.0, 3.0, 5.0, 10.0],
    "YEAR": [2018.0, 2019.0, 2020.0, 2021.0, 2022.0],
    "CRIT_PRICE_WEIGHT": [0.0, 30.0, 50.0, 60.0, 80.0],
    "CRIT_CODE": [0.0, 1.0],
    "TOP_TYPE": ["OPE", "AWP", "NIC", "RES", "NOP", "NOC", "NIP", "COD", "INP"],
    "ISO_COUNTRY_CODE": ["PL", "ES", "FR", "BG", "RO", "SE", "IT", "DK", "DE", "NO"],
    "TYPE_OF_CONTRACT": ["U", "S", "W"],
    "CAE_TYPE": ["6", "3", "1", "4", "8", "R", "N", "5", "Z", "5A"],
    "MAIN_ACTIVITY": [
        "Health", "Defence", "Railway services", "Other",
        "General public\services", "Education", "Environment",
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
    YEAR:              float
    CRIT_PRICE_WEIGHT: float
    CRIT_CODE:         float

    # Categorical
    TOP_TYPE:          str
    ISO_COUNTRY_CODE:  str
    TYPE_OF_CONTRACT:  str
    CAE_TYPE:          str
    MAIN_ACTIVITY:     str

def prepare_input(data: dict) -> pd.DataFrame:
    """Applique le même preprocessing que main.py sur un dict d'input"""
    df = pd.DataFrame([data])
    df['INFO_ON_NON_AWARD'] = 'awarded'
    X, _ = preprocess(df)
    return X

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
        "confidence": round(float(max(proba)), 2)
    }

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
