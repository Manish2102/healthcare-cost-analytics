"""
STEP 6: Serve the trained models behind a real API.

This is what turns "I trained a model in a script" into "I deployed a model" —
the exact JD line ("Engineering: Deployed ML models (API or web app)"). We're
not training anything new here; we load the .joblib files saved by the
previous scripts and wrap them in HTTP endpoints.

Run with:  uvicorn 06_api:app --reload
Then open: http://127.0.0.1:8000/docs  for interactive Swagger docs.
"""
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

HERE = Path(__file__).parent
MODELS = HERE / "models"

app = FastAPI(
    title="Healthcare Cost Analytics API",
    description="Regression, classification, clustering and anomaly detection "
                 "over the public medical-cost dataset.",
)

# Loaded once at startup, not per-request — re-loading a model on every
# request would be needlessly slow.
regression_model = joblib.load(MODELS / "regression_model.joblib")
classification_model = joblib.load(MODELS / "classification_model.joblib")
high_cost_threshold = joblib.load(MODELS / "high_cost_threshold.joblib")
clustering_bundle = joblib.load(MODELS / "clustering_model.joblib")
anomaly_model = joblib.load(MODELS / "anomaly_model.joblib")


class Patient(BaseModel):
    age: int
    sex: Literal["male", "female"]
    bmi: float
    children: int
    smoker: Literal["yes", "no"]
    region: Literal["northeast", "northwest", "southeast", "southwest"]


def to_dataframe(patient: Patient) -> pd.DataFrame:
    return pd.DataFrame([patient.model_dump()])


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict/cost")
def predict_cost(patient: Patient):
    """Regression: predicted medical cost for this patient profile."""
    df = to_dataframe(patient)
    predicted = float(regression_model.predict(df)[0])
    return {"predicted_charges": round(predicted, 2)}


@app.post("/predict/risk")
def predict_risk(patient: Patient):
    """Classification: is this patient high-cost-risk?"""
    df = to_dataframe(patient)
    proba = float(classification_model.predict_proba(df)[0][1])
    return {
        "high_cost_risk": proba >= 0.5,
        "probability": round(proba, 3),
        "threshold_used": round(float(high_cost_threshold), 2),
    }


@app.post("/segment")
def segment(patient: Patient):
    """Clustering: which patient segment does this profile fall into?"""
    df = to_dataframe(patient)
    processed = clustering_bundle["preprocess"].transform(df)
    cluster = int(clustering_bundle["kmeans"].predict(processed)[0])
    return {"cluster": cluster}


@app.post("/predict/anomaly")
def predict_anomaly(patient: Patient, actual_charges: float):
    """
    Anomaly detection: given a patient's ACTUAL charge, is it anomalous
    relative to their profile? (This is the "claims review" use case —
    you already know what was billed; the question is whether it's unusual.)
    """
    df = to_dataframe(patient)
    predicted = float(regression_model.predict(df)[0])
    ratio = actual_charges / predicted if predicted else float("inf")
    return {
        "predicted_charges": round(predicted, 2),
        "actual_charges": actual_charges,
        "ratio": round(ratio, 2),
        "is_anomaly_by_residual": ratio > 2.0,
    }
