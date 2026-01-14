import joblib
import mlflow
import mlflow.sklearn
from pathlib import Path

from sklearn.metrics import roc_auc_score, classification_report
from xgboost import XGBClassifier

PROCESSED_PATH = Path("data/processed")


def load_data():
    X_train = joblib.load(PROCESSED_PATH / "X_train.pkl")
    X_test = joblib.load(PROCESSED_PATH / "X_test.pkl")
    y_train = joblib.load(PROCESSED_PATH / "y_train.pkl")
    y_test = joblib.load(PROCESSED_PATH / "y_test.pkl")
    return X_train, X_test, y_train, y_test


def train_xgboost(params: dict, run_name: str):
    X_train, X_test, y_train, y_test = load_data()

    with mlflow.start_run(run_name=run_name):
        model = XGBClassifier(
            **params,
            objective="binary:logistic",
            eval_metric="auc",
            use_label_encoder=False,
            random_state=42
        )

        model.fit(X_train, y_train)

        y_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_proba >= 0.5).astype(int)

        roc_auc = roc_auc_score(y_test, y_proba)

        print(f"\nRun: {run_name}")
        print("ROC-AUC:", roc_auc)
        print("\nClassification Report:\n")
        print(classification_report(y_test, y_pred))

        mlflow.log_params(params)
        mlflow.log_metric("roc_auc", roc_auc)
        mlflow.sklearn.log_model(model, "model")

        # Save best model explicitly for inference
        Path("artifacts").mkdir(exist_ok=True)
        joblib.dump(model, "artifacts/best_model.pkl")

        return roc_auc, model



def main():
    mlflow.set_experiment("fraud-detection-xgboost")

    best_auc = 0
    best_model = None

    scale_pos_weight = 227845 / 492

    param_grid = [
        {
            "n_estimators": 200,
            "max_depth": 3,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "scale_pos_weight": scale_pos_weight
        },
        {
            "n_estimators": 300,
            "max_depth": 4,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "scale_pos_weight": scale_pos_weight
        },
        {
            "n_estimators": 400,
            "max_depth": 5,
            "learning_rate": 0.03,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "scale_pos_weight": scale_pos_weight
        }
    ]

    for i, params in enumerate(param_grid):
        auc, model = train_xgboost(params, run_name=f"xgboost_run_{i+1}")

        if auc > best_auc:
            best_auc = auc
            best_model = model

    # Save ONLY the best model
    Path("artifacts").mkdir(exist_ok=True)
    joblib.dump(best_model, "artifacts/best_model.pkl")

    print(f"🏆 Best model saved with ROC-AUC: {best_auc:.4f}")


if __name__ == "__main__":
    main()