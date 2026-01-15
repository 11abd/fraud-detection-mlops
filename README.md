**🛡️ Fraud Detection System (MLOps Project)**

A production-ready fraud detection system built end-to-end using Machine Learning and MLOps best practices — from preprocessing and model training to explainability, API deployment, Dockerization, and CI.

**🚀 Overview**

This project addresses a real-world fraud detection problem with:

Extreme class imbalance

Business-driven threshold tuning

Explainable predictions

Deployable inference service

**📊 Dataset**

Credit Card Fraud Dataset (European Cardholders)

~284K transactions

~0.17% fraud cases

PCA-transformed numerical features

Raw data is excluded from the repository.

**🏗️ Architecture**
Preprocessing → XGBoost Model → Threshold Tuning
            → SHAP Explainability → FastAPI API
            → Docker → CI (GitHub Actions)

**⚙️ Tech Stack**

ML: XGBoost, Scikit-learn

Explainability: SHAP

API: FastAPI

MLOps: MLflow, GitHub Actions

Deployment: Docker, Docker Compose

**📁 Project Structure**
src/
 ├── api/              # FastAPI service
 ├── data/             # Preprocessing pipeline
 ├── models/           # Training & evaluation
 ├── explainability/   # SHAP
 └── config/
Dockerfile
docker-compose.yml
.github/workflows/ci.yml

**🎯 Model & Evaluation**

Final model: XGBoost

Imbalance handled using scale_pos_weight

Metrics: ROC-AUC, Recall, Precision

Business-driven threshold tuning (not default 0.5)

**🔍 Explainability**

Global and local feature importance using SHAP

Permutation-based explainer for XGBoost stability

**🌐 Inference API**

POST /predict

Input:

{ "features": [30 numerical values] }


Output:

{
  "fraud_probability": 0.73,
  "fraud_flag": true,
  "threshold_used": 0.3
}

**🐳 Run Locally**
docker build -t fraud-detection-api .
docker compose up


Swagger UI:

http://127.0.0.1:8000/docs

**🔁 CI/CD**

GitHub Actions CI

Validates dependencies

Builds Docker image on every push
**
🧠 Key Highlights**

Business-aligned fraud detection

Explainable ML predictions

Clean separation of code, data, and artifacts

Dockerized & CI-enabled system

**👤 Author**

Abd
Machine Learning / MLOps Engineer
