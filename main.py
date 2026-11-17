"""Command-line entry point for the fraud comparison experiment."""
import argparse
import importlib.util
import logging
from pathlib import Path

from src.utils import DATASET, ROOT, read_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["train"], default="train")
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--data", type=Path, default=DATASET)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--mlflow", action="store_true")
    args = parser.parse_args()
    if args.mlflow and importlib.util.find_spec("mlflow") is None:
        parser.error("Install mlflow before using --mlflow.")
    from src.train import run

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    profile = "fast" if args.fast else "full"
    config = read_json(args.config or ROOT / "configs" / f"{profile}.json")
    try:
        run(args.data, config, profile, args.mlflow)
    except (FileNotFoundError, ValueError) as exc:
        parser.exit(2, f"{exc}\n")


if __name__ == "__main__":
    main()
