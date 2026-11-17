"""Validate input without silently replacing invalid or missing data."""
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils import DATASET

FEATURES = ["Time", *(f"V{i}" for i in range(1, 29)), "Amount"]
TARGET = "Class"
SOURCE = "https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud"


def features(frame: pd.DataFrame, allow_missing: bool = False) -> pd.DataFrame:
    if frame.empty:
        raise ValueError("At least one transaction is required.")
    missing = set(FEATURES) - set(frame)
    extra = set(frame) - set(FEATURES)
    if missing or extra or frame.columns.has_duplicates:
        raise ValueError(f"Invalid feature schema. Missing: {sorted(missing)}. Extra: {sorted(extra)}.")
    try:
        result = frame[FEATURES].apply(pd.to_numeric, errors="raise").astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError("All features must be numeric.") from exc
    if np.isinf(result.to_numpy()).any() or (not allow_missing and result.isna().any().any()):
        raise ValueError("Features must be finite numbers.")
    if (result[["Time", "Amount"]] < 0).any().any():
        raise ValueError("Time and Amount must be nonnegative.")
    return result


def validate_dataset(frame: pd.DataFrame) -> pd.DataFrame:
    if TARGET not in frame or frame.columns.has_duplicates:
        raise ValueError("Expected one Class column and unique column names.")
    x = features(frame.drop(columns=TARGET), allow_missing=True)
    if x.isna().all().any():
        raise ValueError("A feature contains only missing values.")
    y = pd.to_numeric(frame[TARGET], errors="raise")
    if y.isna().any() or set(y) != {0, 1}:
        raise ValueError("Class must contain both 0 and 1, without missing labels.")
    return x.assign(Class=y.astype(int))


def load_dataset(path: Path = DATASET) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Download {SOURCE} and extract creditcard.csv to {path}.")
    return validate_dataset(pd.read_csv(path))
