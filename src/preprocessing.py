"""Partition records and build pipelines that fit exclusively on training rows."""
import pandas as pd
from imblearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.data_loader import FEATURES, TARGET


def split_data(frame: pd.DataFrame, test_size: float, seed: int):
    # Identical observed inputs can cross partitions even when their labels differ.
    # Keep the first exact duplicate, and exclude groups with conflicting labels.
    unique = frame.drop_duplicates()
    conflicting = unique.duplicated(FEATURES, keep=False)
    clean = unique.loc[~conflicting]
    train, test = train_test_split(
        clean, test_size=test_size, random_state=seed, stratify=clean[TARGET]
    )
    return train, test, {"exact_duplicates_removed": len(frame) - len(unique),
                         "conflicting_rows_removed": int(conflicting.sum())}


def stratified_subset(frame, limit, seed):
    if limit is None or len(frame) <= limit:
        return frame
    subset, _ = train_test_split(frame, train_size=limit, stratify=frame[TARGET], random_state=seed)
    return subset


def pipeline(model, scale=True):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scale", StandardScaler() if scale else "passthrough"),
        ("sample", "passthrough"),
        ("model", model),
    ])
