# Credit Card Fraud Detection

A FastAPI web application that demonstrates credit-card fraud classification with an adjustable decision threshold.

## Features
- Pre-trained Logistic Regression pipeline
- Adjustable classification threshold from 0.00 to 1.00
- Accuracy, Precision, Recall and F1-Score
- Confusion Matrix
- Responsive HTML/CSS/JavaScript interface

## Project Structure

```text
Credit_Card_Fraud_Detection/
├── main.py
├── requirements.txt
├── render.yaml
├── .gitignore
├── model/
│   ├── fraud_model.pkl
│   └── evaluation_data.pkl
├── templates/
│   └── index.html
└── static/
    ├── script.js
    └── style.css
```

## Run Locally

```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

Open `http://127.0.0.1:8000`.

## Deployment

This project is configured for Render with `render.yaml`.

The dataset and virtual environment are intentionally excluded because the deployed application uses the already-trained model and saved evaluation probabilities.
