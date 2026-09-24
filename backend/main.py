import joblib
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

app = FastAPI(
    title="Credit Card Fraud Detection API",
    description="Fraud detection API with an adjustable classification threshold",
)

# Allow the separately deployed frontend to communicate with this backend.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load trained model and evaluation data
model = joblib.load("model/fraud_model.pkl")
evaluation_data = joblib.load("model/evaluation_data.pkl")

y_test = evaluation_data["y_test"]
probabilities = evaluation_data["probabilities"]


@app.get("/")
def home():
    return {
        "message": "Credit Card Fraud Detection API is running",
        "docs": "/docs",
    }


@app.get("/api/evaluate")
def evaluate_threshold(threshold: float = 0.50):
    threshold = max(0.0, min(1.0, threshold))

    predictions = (probabilities >= threshold).astype(int)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(
        y_test,
        predictions,
        zero_division=0,
    )
    recall = recall_score(
        y_test,
        predictions,
        zero_division=0,
    )
    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0,
    )
    matrix = confusion_matrix(y_test, predictions)

    return {
        "threshold": threshold,
        "accuracy": round(accuracy * 100, 2),
        "precision": round(precision * 100, 2),
        "recall": round(recall * 100, 2),
        "f1_score": round(f1 * 100, 2),
        "confusion_matrix": matrix.tolist(),
    }