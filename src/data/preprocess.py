import yaml
import joblib
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.data.load_data import load_raw_data


def load_config(config_path: str):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def preprocess_and_split(config_path: str):
    config = load_config(config_path)

    # Load data
    df = load_raw_data(config["data"]["raw_path"])

    target = config["features"]["target"]
    drop_cols = config["features"]["drop_columns"]

    if drop_cols:
        df = df.drop(columns=drop_cols)

    X = df.drop(columns=[target])
    y = df[target]

    # Train-test split (STRATIFIED)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config["split"]["test_size"],
        random_state=config["split"]["random_state"],
        stratify=y
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Save processed artifacts
    processed_path = Path(config["data"]["processed_path"])
    processed_path.mkdir(parents=True, exist_ok=True)

    joblib.dump(X_train_scaled, processed_path / "X_train.pkl")
    joblib.dump(X_test_scaled, processed_path / "X_test.pkl")
    joblib.dump(y_train, processed_path / "y_train.pkl")
    joblib.dump(y_test, processed_path / "y_test.pkl")
    joblib.dump(scaler, processed_path / "scaler.pkl")

    print("Preprocessing completed successfully")
    print(f"Train shape: {X_train_scaled.shape}")
    print(f"Test shape : {X_test_scaled.shape}")


if __name__ == "__main__":
    preprocess_and_split("src/config/config.yaml")
