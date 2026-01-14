import joblib
import numpy as np
from pathlib import Path

from sklearn.metrics import (
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix
)

PROCESSED_PATH = Path("data/processed")
MODEL_PATH = Path("mlruns")


def load_data():
    X_test = joblib.load(PROCESSED_PATH / "X_test.pkl")
    y_test = joblib.load(PROCESSED_PATH / "y_test.pkl")
    return X_test, y_test


def load_model():
    model_path = Path("artifacts/best_model.pkl")
    if not model_path.exists():
        raise FileNotFoundError("Best model not found. Train model first.")
    return joblib.load(model_path)



def evaluate_thresholds():
    X_test, y_test = load_data()
    model = load_model()

    y_proba = model.predict_proba(X_test)[:, 1]

    print("\nThreshold | Recall | Precision | False Negatives")
    print("-" * 55)

    for threshold in np.arange(0.1, 0.91, 0.1):
        y_pred = (y_proba >= threshold).astype(int)

        recall = recall_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)

        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

        print(
            f"{threshold:.2f}      | "
            f"{recall:.3f}  | "
            f"{precision:.3f}    | "
            f"{fn}"
        )

def business_cost(fn, fp, cost_fn=1000, cost_fp=10):
    return fn * cost_fn + fp * cost_fp


if __name__ == "__main__":
    evaluate_thresholds()
