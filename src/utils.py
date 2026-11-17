"""Paths and JSON serialization shared by training and serving."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data/raw/creditcard.csv"
MODELS = ROOT / "models"


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))
