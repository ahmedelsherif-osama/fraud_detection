# Fraud Detection Pipeline

## Overview

This project implements an end-to-end **fraud detection system** that predicts whether a financial transaction is fraudulent.

The pipeline covers:

- Data ingestion & preprocessing  
- Feature transformation  
- Model training (XGBoost)  
- Evaluation  
- Inference (CLI + API-ready)  

---

## Problem

Given a transaction with anonymized features:

{
  "Time": 10000,
  "V1": -1.35,
  ...
  "Amount": 149.62
}

The model predicts:

{
  "fraud_probability": 0.87,
  "fraud_prediction": 1
}

Where:

- `1` → Fraud  
- `0` → Legitimate  

---

## Pipeline

ingest_data()
→ split_dataset()
→ build_features()
→ train_model()
→ train_final()
→ evaluate_model()

---

## Model

- Algorithm: **XGBoost (Gradient Boosting)**
- Task: Binary classification (Fraud vs Non-Fraud)
- Output: Probability + threshold-based classification

---

## Inference

- Loads latest trained model
- Applies preprocessing (scaling)
- Returns probability + prediction

---

## CLI Usage

python predict.py --input data/sample.csv --threshold 0.5

---

## Artifacts

artifacts/
- xgb_final_model_*.joblib
- amount_scaler.joblib

---

## Deployment

- FastAPI
- Docker
- Ready for cloud deployment

---

## Note

Models are stored in-repo for simplicity.  
In production, use object storage or model registry.

---

## Key Learnings

- Data preprocessing consistency is critical  
- Class imbalance affects performance  
- Threshold tuning impacts precision/recall  

---

## Tech Stack

- Python  
- XGBoost  
- scikit-learn  
- pandas / numpy  
- FastAPI  
- Docker  

---

## Author Notes

Focus on:
- clean ML pipelines  
- reproducibility  
- deployment readiness  