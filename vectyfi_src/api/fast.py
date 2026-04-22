from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np
import pandas as pd
import random

app = FastAPI()

try:
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
except FileNotFoundError:
    model = None


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

@app.get("/")
def root():
    return {"message": "Vectyfi, Tender Prediction API!"}

@app.post("/predict")
def predict(data: TenderInput):
    if model is None:
        return {"error": "Model not available yet"}
    df = pd.DataFrame([data.model_dump()])
    prediction = model.predict(df)
    return {"prediction": int(prediction[0])}


# To launch server : uvicorn fast:app --reload
# Request with http://localhost:8000/docs

DUMMY_VALUES = {
    "B_MULTIPLE_CAE": ["Y", "N"], "B_EU_FUNDS": ["Y", "N"],
    "B_GPA": ["Y", "N"], "B_FRA_AGREEMENT": ["Y", "N"], "B_ACCELERATED": ["Y", "N"],
    "LOTS_NUMBER": [1.0, 3.0, 5.0, 10.0],
    "YEAR": [2018.0, 2019.0, 2020.0, 2021.0, 2022.0],
    "CRIT_PRICE_WEIGHT": [0.0, 30.0, 50.0, 60.0, 80.0],
    "CRIT_CODE": [0.0, 1.0],
    "TOP_TYPE": ["OPEN", "RESTRICTED", "NEGOTIATED"],
    "ISO_COUNTRY_CODE": ["FR", "DE", "BE", "IT", "ES"],
    "TYPE_OF_CONTRACT": ["WORKS", "SUPPLIES", "SERVICES"],
    "CAE_TYPE": ["1", "2", "3", "4"],
    "MAIN_ACTIVITY": ["GENERAL", "EDUCATION", "HEALTH"],
}

@app.get("/dummy/predict")
def dummy_predict():
    dummy = {key: random.choice(values) for key, values in DUMMY_VALUES.items()}
    df = pd.DataFrame([dummy])
    prediction = model.predict(df)
    proba = model.predict_proba(df)[0]
    return {
        "input": dummy,
        "accepted": bool(prediction[0]),
        "confidence": round(float(max(proba)), 2)
    }
