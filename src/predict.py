"""Serve only pipelines listed in the local, versioned training registry."""
import re
from functools import lru_cache
from importlib.metadata import version
from pathlib import Path

import joblib
import pandas as pd

from src.data_loader import FEATURES, features
from src.model_selection import fraud_scores
from src.utils import MODELS, read_json

MAX_BATCH = 10_000


class ArtifactsUnavailable(Exception):
    """No usable local model registry is available."""


def registry(root: Path = MODELS):
    try:
        run_id = read_json(root / "latest.json")["run_id"]
        if not re.fullmatch(r"\d{8}T\d{12}Z", run_id):
            raise ValueError("Invalid run identifier")
        directory = root / run_id
        metadata = read_json(directory / "metadata.json")
        if metadata["features"] != FEATURES or metadata["run_id"] != run_id:
            raise ValueError("Model feature schema or run identifier differs")
        return directory, metadata
    except FileNotFoundError as exc:
        raise ArtifactsUnavailable("Run python main.py --mode train to generate model artifacts.") from exc
    except (ValueError, KeyError, TypeError) as exc:
        raise ArtifactsUnavailable("Model registry is invalid. Re-run training.") from exc


@lru_cache(maxsize=8)
def _load(path: Path, modified_ns: int):
    return joblib.load(path)


def predict(frame: pd.DataFrame, model: str | None = None, root: Path = MODELS):
    x = features(frame)
    if len(x) > MAX_BATCH:
        raise ValueError(f"Each request may contain up to {MAX_BATCH:,} transactions.")
    directory, metadata = registry(root)
    name = model if model is not None else metadata["best_model"]
    if name not in metadata["models"] or not re.fullmatch(r"[a-z][a-z0-9_]*", name):
        raise ValueError(f"Unknown model: {name}")
    for dependency, trained_version in metadata["packages"].items():
        if dependency in {"numpy", "pandas", "scipy", "scikit-learn", "imbalanced-learn",
                          "xgboost", "joblib"} and version(dependency) != trained_version:
            raise ArtifactsUnavailable(f"Model requires {dependency}=={trained_version}.")
    path = directory / f"{name}.joblib"
    try:
        estimator = _load(path, path.stat().st_mtime_ns)
    except (OSError, ValueError, EOFError) as exc:
        raise ArtifactsUnavailable("Model artifact is missing or unreadable.") from exc
    scores, kind = fraud_scores(estimator, x)
    threshold = metadata["models"][name]["threshold"]
    return pd.DataFrame({
        "prediction": (scores >= threshold).astype(int),
        "fraud_score": scores,
        "score_kind": kind,
        "model": name,
        "threshold": threshold,
        "run_id": metadata["run_id"],
    }, index=x.index)


def predict_csv(file, model=None, root=MODELS):
    frame = pd.read_csv(file, nrows=MAX_BATCH + 1)
    predictions = predict(frame, model, root)
    return pd.concat([frame, predictions], axis=1)
