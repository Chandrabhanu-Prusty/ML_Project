"""Run reproducible experiments and publish a complete versioned model registry."""
import hashlib
import logging
import platform
from datetime import datetime, timezone
from importlib.metadata import version
from tempfile import NamedTemporaryFile
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GridSearchCV, ParameterGrid, StratifiedKFold
from threadpoolctl import threadpool_limits

from src.data_loader import FEATURES, TARGET, load_dataset
from src.evaluate import evaluate
from src.model_selection import candidates, fraud_scores, select_threshold
from src.preprocessing import pipeline, split_data, stratified_subset
from src.utils import MODELS, ROOT, write_json
from src.visualization import comparison, eda, embeddings, insights

LOG = logging.getLogger(__name__)


def learning_curves(frame, cv, models, seed):
    rows = {}
    for name, estimator in models.items():
        LOG.info("Learning curve: %s", name)
        grouped = []
        for fraction in (0.2, 0.5, 1.0):
            observed = []
            for tr, va in list(cv.split(frame, frame[TARGET]))[:3]:
                fold = frame.iloc[tr]
                validation = frame.iloc[va]
                limit = min(len(fold), max(500, round(len(fold) * fraction)))
                sample = stratified_subset(fold, limit, seed)
                fitted = clone(estimator)
                sampler = fitted.named_steps["sample"]
                if sampler != "passthrough":
                    positives = int(sample.Class.sum())
                    sampler.set_params(k_neighbors=min(sampler.k_neighbors, positives - 1))
                fitted.fit(sample[FEATURES], sample[TARGET])
                scores = []
                for data in (sample, validation):
                    score, _ = fraud_scores(fitted, data[FEATURES])
                    scores.append(average_precision_score(data[TARGET], score))
                observed.append([len(sample), *scores])
            values = np.array(observed)
            grouped.append({
                "rows": round(values[:, 0].mean()),
                "train_ap": float(values[:, 1].mean()),
                "validation_ap": float(values[:, 2].mean()),
                "validation_sd": float(values[:, 2].std()),
            })
        rows[name] = grouped
    return rows


def trial_grid(name, grid):
    if name in {"logistic", "knn", "svm", "mlp"}:
        return grid
    # Fractional designs cover all requested parameter levels with a limited budget.
    indices = {"tree": [0, 3, 5, 6, 8, 11], "xgboost": [0, 3, 5, 6],
               "forest": [0, 3, 5, 6]}[name]
    combinations = list(ParameterGrid(grid))
    return [{key: [value] for key, value in combinations[i].items()} for i in indices]


def run(dataset, config, profile, use_mlflow=False):
    frame = load_dataset(dataset)
    train, test, exclusions = split_data(frame, config["test_size"], config["seed"])
    train = stratified_subset(train, config["development_rows"], config["seed"])
    # Threshold validation is disjoint from CV, including hyperparameter selection.
    fit, threshold_data, _ = split_data(train, 0.2, config["seed"] + 1)
    folds = config["folds"]
    counts = fit.Class.value_counts()
    if counts.min() < 2 * folds:
        raise ValueError(f"Need at least {2 * folds} rows of each class in the fitting partition.")
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=config["seed"])
    min_positive = int(counts[1] - np.ceil(counts[1] / folds))
    ratio = max(
        fit.Class.iloc[i].mean() / (1 - fit.Class.iloc[i].mean())
        for i, _ in cv.split(fit, fit.Class)
    )
    ratio = max(ratio, counts[1] / counts[0])
    if ratio >= 1:
        raise ValueError("The dataset must have fewer fraud than legitimate transactions.")
    specs = candidates(config, min_positive, ratio)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    models_dir = MODELS / run_id
    report_dir = ROOT / "reports/runs" / run_id
    figures = report_dir / "figures"
    models_dir.mkdir(parents=True)
    figures.mkdir(parents=True)
    summary = eda(frame, figures, report_dir)
    LOG.info("Observed %s rows, %s fraud (%.4f%%)", summary["rows"], summary["fraud"],
             summary["fraud_percent"])
    examples = test.groupby(TARGET, group_keys=False).head(3)
    examples.to_csv(report_dir / "held_out_examples.csv", index=False)
    test_x, test_y = test[FEATURES], test[TARGET]

    metrics, curves, fitted, importance = {}, {}, {}, {}
    selections = {}
    with threadpool_limits(config["threads"]):
        for name, (model, scale, grid) in specs.items():
            data = stratified_subset(fit, config.get(f"{name}_rows"), config["seed"])
            LOG.info("Training %s on %s rows", name, len(data))
            search = GridSearchCV(pipeline(model, scale), trial_grid(name, grid),
                                  scoring="average_precision", cv=cv, n_jobs=1,
                                  return_train_score=True, error_score="raise")
            start = perf_counter()
            search.fit(data[FEATURES], data[TARGET])
            duration = perf_counter() - start
            pd.DataFrame(search.cv_results_).to_csv(report_dir / f"cv_{name}.csv", index=False)
            estimator = search.best_estimator_
            fitted[name] = estimator
            selections[name] = {
                "cv_average_precision": float(search.best_score_),
                "cv_sd": float(search.cv_results_["std_test_score"][search.best_index_]),
                "fit_rows": len(data),
                "parameters": {key: repr(value) for key, value in search.best_params_.items()},
                "training_seconds": duration,
            }

        # Rank before evaluating the holdout. The smaller SVM CV set has a
        # different sampling variance, so use the common threshold validation
        # partition for ranking every chosen configuration.
        for name, estimator in fitted.items():
            score, _ = fraud_scores(estimator, threshold_data[FEATURES])
            selections[name]["selection_average_precision"] = float(
                average_precision_score(threshold_data[TARGET], score)
            )
            selections[name]["threshold"] = select_threshold(threshold_data[TARGET], score)
        best = max(selections, key=lambda n: selections[n]["selection_average_precision"])
        for name, estimator in fitted.items():
            threshold = selections[name]["threshold"]
            metrics[name], curves[name] = evaluate(estimator, test_x, test_y, threshold)
            metrics[name].update(selections[name])
            joblib.dump(estimator, models_dir / f"{name}.joblib", compress=3)

        # Use a documented diagnostic cap so the repeated forest fits remain practical.
        diagnostic = stratified_subset(fit, 30000, config["seed"])
        learning = learning_curves(diagnostic, cv, {
            name: fitted[name] for name in ("logistic", "tree", "forest", "mlp")
        }, config["seed"])
        reduction = embeddings(fit, config["embedding_rows"], config["seed"], figures)

    for name, attr in [("logistic", "coef_"), ("tree", "feature_importances_"),
                       ("forest", "feature_importances_")]:
        values = np.asarray(getattr(fitted[name].named_steps["model"], attr)).ravel()
        importance[name] = [{"feature": key, "value": float(value)}
                            for key, value in zip(FEATURES, values)]
    loss = fitted["mlp"].named_steps["model"].loss_curve_
    comparison(metrics, curves, figures)
    insights(learning, importance, loss, figures)
    with dataset.open("rb") as file:
        digest = hashlib.file_digest(file, "sha256").hexdigest()
    metadata = {
        "run_id": run_id, "profile": profile, "best_model": best,
        "dataset_sha256": digest, "features": FEATURES, "dataset": summary,
        "exclusions": exclusions, "configuration": config,
        "fit_rows": len(fit), "threshold_rows": len(threshold_data),
        "threshold_fraud": int(threshold_data.Class.sum()),
        "test_rows": len(test), "test_fraud": int(test.Class.sum()),
        "diagnostic_rows": len(diagnostic), "embedding": reduction,
        "python": platform.python_version(),
        "packages": {name: version(name) for name in (
            "numpy", "pandas", "scipy", "scikit-learn", "imbalanced-learn",
            "xgboost", "joblib", "matplotlib", "seaborn",
        )},
        "models": metrics,
    }
    write_json(models_dir / "metadata.json", metadata)
    write_json(report_dir / "metadata.json", metadata)
    pd.DataFrame(metrics).T.rename_axis("model").to_csv(report_dir / "model_comparison.csv")
    write_json(report_dir / "curves.json", curves)
    write_json(report_dir / "insights.json", {
        "learning": learning, "importance": importance, "mlp_loss": loss,
    })
    if use_mlflow:
        import mlflow

        with mlflow.start_run(run_name=run_id):
            mlflow.log_params({"profile": profile, "best_model": best, "sha256": digest})
            mlflow.log_metrics({f"{key}.test_ap": value["average_precision"]
                                for key, value in metrics.items()})
            mlflow.log_artifacts(str(report_dir))
    with NamedTemporaryFile(dir=MODELS, delete=False) as pointer:
        temporary = MODELS / pointer.name
    write_json(temporary, {"run_id": run_id})
    temporary.replace(MODELS / "latest.json")
    LOG.info("Selected %s; artifacts at %s", best, report_dir)
    return metadata
