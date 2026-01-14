import joblib
import shap
import numpy as np
import pandas as pd
from pathlib import Path

PROCESSED_PATH = Path("data/processed")
ARTIFACTS_PATH = Path("artifacts")


def load_artifacts():
    model = joblib.load(ARTIFACTS_PATH / "best_model.pkl")
    X_test = joblib.load(PROCESSED_PATH / "X_test.pkl")

    # Convert to DataFrame for SHAP stability
    X_test = pd.DataFrame(X_test)
    return model, X_test


def explain_global():
    model, X_test = load_artifacts()

    # ✅ Universal SHAP explainer (robust)
    explainer = shap.Explainer(
        model.predict_proba,
        X_test,
        algorithm="permutation"
    )

    shap_values = explainer(X_test[:300])

    shap.summary_plot(
        shap_values[:, :, 1],  # class 1 = fraud
        X_test[:300],
        show=True
    )


def explain_single(index=0):
    model, X_test = load_artifacts()

    explainer = shap.Explainer(
        model.predict_proba,
        X_test,
        algorithm="permutation"
    )

    shap_values = explainer(X_test[index:index + 1])

    shap.force_plot(
        explainer.expected_value[1],
        shap_values.values[:, :, 1],
        X_test.iloc[index],
        matplotlib=True
    )


if __name__ == "__main__":
    explain_global()
