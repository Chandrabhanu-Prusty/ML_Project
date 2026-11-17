"""Publish selected run evidence into the Markdown report and figures directory."""
from shutil import copyfile

import pandas as pd

from src.predict import registry
from src.utils import ROOT, read_json

FIGURES = {
    "class_distribution": "Observed class distribution on a logarithmic scale.",
    "amount_time": "Amount and elapsed time distributions.",
    "class_features": "Amount and two selected PCA-derived features by class.",
    "correlation": "Pearson correlations in the observed dataset.",
    "comparison": "Holdout AP, model search and fit time, and holdout scoring time.",
    "curves": "Holdout ROC and precision–recall curves from continuous scores.",
    "confusion_matrices": "Holdout confusion matrices at validation-selected thresholds.",
    "learning_curves": "Average precision against training size. Validation bars indicate fold standard deviation.",
    "importance": "Logistic coefficients and tree impurity importance for the ten largest values.",
    "mlp_loss": "Selected MLP's training loss across epochs.",
    "pca_variance": "Per-component and cumulative explained variance of training PCA.",
    "pca": "Two-component PCA projection of the fraud-enriched training sample.",
    "tsne": "t-SNE embedding of the same training sample.",
}


def table(headers, rows):
    return "\n".join([
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
        *["| " + " | ".join(str(value) for value in row) + " |" for row in rows],
    ])


def fmt(value):
    return f"{value:.4f}"


def publish():
    _, meta = registry()
    run = ROOT / "reports/runs" / meta["run_id"]
    figures = ROOT / "reports/figures"
    figures.mkdir(parents=True, exist_ok=True)
    for name in FIGURES:
        copyfile(run / "figures" / f"{name}.png", figures / f"{name}.png")
    models = meta["models"]
    winner = meta["best_model"]
    selected = models[winner]
    test_leader = max(models, key=lambda name: models[name]["average_precision"])
    data = meta["dataset"]
    exclusions = meta["exclusions"]
    holdout = meta["test_fraud"]
    config = meta["configuration"]
    heading = f"""# Measured experiment results

Run `{meta['run_id']}` used the **{meta['profile']}** profile, seed {config['seed']},
and {config['folds']} stratified CV folds. Python {meta['python']} and pinned package
versions are recorded in the local run metadata. Dataset SHA-256 is
`{meta['dataset_sha256']}`.

## Observed dataset and partitions

The source CSV contains **{data['rows']:,} transactions**, with {data['fraud']:,} fraud
and {data['legitimate']:,} legitimate transactions ({data['fraud_percent']:.4f}% fraud).
It has {sum(data['missing'].values()):,} missing feature cells and {data['duplicate_rows']:,}
exact duplicates. Modeling collapsed {exclusions['exact_duplicates_removed']:,} duplicate
rows and excluded {exclusions['conflicting_rows_removed']:,} rows with contradictory labels at
identical feature values. The common fitting pool had {meta['fit_rows']:,} rows and the
threshold validation pool had {meta['threshold_rows']:,} rows with {meta['threshold_fraud']} fraud.
The holdout had {meta['test_rows']:,} rows with {holdout} fraud, an all-legitimate
accuracy of {1 - holdout / meta['test_rows']:.4%} and a no-skill AP of
{holdout / meta['test_rows']:.5f}.

## Model comparison

**{winner}** led common validation AP ({selected['selection_average_precision']:.4f}).
Its holdout AP was **{selected['average_precision']:.4f}**. At its validation-selected
threshold, it caught **{holdout - selected['false_negatives']} of {holdout}**
holdout fraud cases and flagged **{selected['false_positives']} legitimate transactions**.
The highest observed holdout AP belonged to **{test_leader}**
({models[test_leader]['average_precision']:.4f}). Model deployment selection was fixed
before evaluating holdout scores.

Random forest and XGBoost reached holdout AP of {models['forest']['average_precision']:.4f}
and {models['xgboost']['average_precision']:.4f}. At their own validation-selected
thresholds, they flagged {models['forest']['false_positives']} and
{models['xgboost']['false_positives']} legitimate transactions, respectively.

AP, ROC AUC and CV AP use continuous scores. The other rates use the stored threshold.
SVM thresholds refer to signed margins. Training seconds include CV and the final refit.
Scoring time per row is normalized from one complete holdout batch.
"""
    if meta["profile"] == "fast":
        heading += "\nThe development profile uses fewer rows and folds. Treat these measurements as a workflow check.\n"
    performance = table(
        ["Model", "Fit rows", "CV AP ± SD", "Validation AP", "Holdout AP", "ROC AUC",
         "Accuracy", "Precision", "Recall", "F1"],
        [[name, f"{m['fit_rows']:,}", f"{m['cv_average_precision']:.4f} ± {m['cv_sd']:.4f}",
          *[fmt(m[key]) for key in ("selection_average_precision", "average_precision", "roc_auc",
                                   "accuracy", "precision", "recall", "f1")]]
         for name, m in models.items()],
    )
    costs = table(
        ["Model", "Threshold", "False positives", "False negatives", "FPR", "FNR",
         "Search + fit seconds", "Scoring μs/row"],
        [[name, fmt(m["threshold"]), m["false_positives"], m["false_negatives"],
          f"{m['false_positive_rate']:.6f}", fmt(m["false_negative_rate"]),
          f"{m['training_seconds']:.2f}", f"{m['inference_microseconds_per_row']:.2f}"]
         for name, m in models.items()],
    )

    insights = read_json(run / "insights.json")
    learn = table(
        ["Model", "Rows/fold", "Train AP", "Validation AP", "Train − validation",
         "Validation AP change from smallest size"],
        [[name, row["rows"], fmt(row["train_ap"]), fmt(row["validation_ap"]),
          fmt(row["train_ap"] - row["validation_ap"]),
          f"{row['validation_ap'] - points[0]['validation_ap']:+.4f}"]
         for name, points in insights["learning"].items() for row in [points[-1]]],
    )
    final = {name: points[-1] for name, points in insights["learning"].items()}
    largest_gap = max(final, key=lambda name: final[name]["train_ap"] - final[name]["validation_ap"])
    logistic = pd.read_csv(run / "cv_logistic.csv").fillna({"param_model__class_weight": "none"})
    logistic["Treatment"] = [
        "SMOTE" if str(sample).startswith("SMOTE") else
        "Class weighting" if weight == "balanced" else "Original"
        for sample, weight in zip(logistic.param_sample, logistic.param_model__class_weight)
    ]
    regularization = table(
        ["Penalty", "C", "Imbalance treatment", "CV train AP", "CV validation AP", "CV SD"],
        [["L1" if row.param_model__l1_ratio == 1 else "L2", row.param_model__C,
          row.Treatment, fmt(row.mean_train_score), fmt(row.mean_test_score),
          fmt(row.std_test_score)]
         for row in logistic.sort_values("mean_test_score", ascending=False).itertuples()],
    )
    embedding = meta["embedding"]
    discussion = f"""## Learning diagnostics and regularization

The following rows give the largest training size in each diagnostic curve, with the change
in validation AP from the smallest size. The separate diagnostic pool contains
{meta['diagnostic_rows']:,} rows.

{learn}

At the largest diagnostic size, **{largest_gap}** had the widest observed train–validation
AP gap, consistent with greater sensitivity to the fitted sample. Curve shapes help
distinguish persistent gaps from gaps that shrink with added data. Rare fraud and limited
fold counts make the curve differences noisy.

Logistic CV varies penalty, strength and imbalance policy. Compare cells with the same
treatment and C to isolate the penalty comparison.

{regularization}

The MLP logged {len(insights['mlp_loss'])} training epochs. Training loss fell from
{insights['mlp_loss'][0]:.4f} to {insights['mlp_loss'][-1]:.4f}.
PCA's first two components explain {embedding['pca_first_two_variance']:.2%} of standardized
fitting-partition variance. Both scatterplots use {embedding['rows']:,} observations,
including {embedding['fraud']:,} fraud cases. Enrichment makes these points visible.

## Figures
"""
    images = "\n\n".join(
        f"![{label}](figures/{name}.png)\n\n*Figure {i}. {label}*"
        for i, (name, label) in enumerate(FIGURES.items(), 1)
    )
    (ROOT / "reports/results.md").write_text(
        "\n\n".join([heading, performance, costs, discussion, images]) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    publish()
