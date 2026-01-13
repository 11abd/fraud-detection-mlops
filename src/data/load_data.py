import pandas as pd
from pathlib import Path


def load_raw_data(path: str) -> pd.DataFrame:
    """
    Load raw CSV data
    """
    if not Path(path).exists():
        raise FileNotFoundError(f"Data not found at {path}")
    return pd.read_csv(path)
