from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"

app = FastAPI(
    title="Credit Card Fraud Detection",
    description="Fraud detection with an adjustable classification threshold",
)

model = joblib.load(MODEL_DIR / "fraud_model.pkl")
evaluation_data = joblib.load(MODEL_DIR / "evaluation_data.pkl")

y_test = evaluation_data["y_test"]
probabilities = evaluation_data["probabilities"]

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", response_class=HTMLResponse)
def home():
    return (TEMPLATES_DIR / "index.html").read_text(encoding="utf-8")


@app.get("/api/evaluate")
def evaluate_threshold(threshold: float = 0.50):
    threshold = max(0.0, min(1.0, threshold))
    predictions = (probabilities >= threshold).astype(int)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, zero_division=0)
    recall = recall_score(y_test, predictions, zero_division=0)
    f1 = f1_score(y_test, predictions, zero_division=0)
    matrix = confusion_matrix(y_test, predictions)

    return {
        "threshold": threshold,
        "accuracy": round(accuracy * 100, 2),
        "precision": round(precision * 100, 2),
        "recall": round(recall * 100, 2),
        "f1_score": round(f1 * 100, 2),
        "confusion_matrix": matrix.tolist(),
    }
