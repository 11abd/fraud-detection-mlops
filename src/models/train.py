import joblib
import mlflow
import mlflow.sklearn
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    precision_recall_curve
)

PROCESSED_PATH = Path("data/processed")


def load_data():
    X_train = joblib.load(PROCESSED_PATH / "X_train.pkl")
    X_test = joblib.load(PROCESSED_PATH / "X_test.pkl")
    y_train = joblib.load(PROCESSED_PATH / "y_train.pkl")
    y_test = joblib.load(PROCESSED_PATH / "y_test.pkl")
    return X_train, X_test, y_train, y_test


def train_baseline():
    X_train, X_test, y_train, y_test = load_data()

    mlflow.set_experiment("fraud-detection-baseline")

    with mlflow.start_run(run_name="logistic_regression_baseline"):
        model = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42
        )

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        roc_auc = roc_auc_score(y_test, y_proba)

        print("ROC-AUC:", roc_auc)
        print("\nClassification Report:\n")
        print(classification_report(y_test, y_pred))

        # Log params & metrics
        mlflow.log_param("model", "LogisticRegression")
        mlflow.log_param("class_weight", "balanced")
        mlflow.log_metric("roc_auc", roc_auc)

        # Save model
        model_path = "model"
        mlflow.sklearn.log_model(model, model_path)

        print("Model training logged to MLflow")


if __name__ == "__main__":
    train_baseline()
