import joblib
import yaml
import numpy as np
from pathlib import Path
from fastapi import FastAPI
from pydantic import BaseModel

# Paths
ARTIFACTS_PATH = Path("artifacts")
PROCESSED_PATH = Path("data/processed")
CONFIG_PATH = Path("src/config/threshold.yaml")

# Load artifacts at startup
model = joblib.load(ARTIFACTS_PATH / "best_model.pkl")
scaler = joblib.load(PROCESSED_PATH / "scaler.pkl")

with open(CONFIG_PATH, "r") as f:
    threshold = yaml.safe_load(f)["decision_threshold"]

app = FastAPI(
    title="Fraud Detection API",
    description="Production-grade fraud detection service",
    version="1.0"
)


class Transaction(BaseModel):
    features: list[float]


@app.get("/")
def health_check():
    return {"status": "ok"}


@app.post("/predict")
def predict(transaction: Transaction):
    X = np.array(transaction.features).reshape(1, -1)
    X_scaled = scaler.transform(X)

    fraud_proba = model.predict_proba(X_scaled)[0][1]
    fraud_flag = fraud_proba >= threshold

    return {
        "fraud_probability": round(float(fraud_proba), 4),
        "fraud_flag": bool(fraud_flag),
        "threshold_used": threshold
    }
