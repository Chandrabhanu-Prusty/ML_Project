"""Reproducible report figures and tabular diagnostics."""
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from src.data_loader import FEATURES
from src.utils import write_json

sns.set_theme(style="whitegrid", palette=["#247e83", "#db7848"])


def save(fig, directory, name):
    fig.tight_layout()
    fig.savefig(directory / f"{name}.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def eda(frame, directory, tables):
    summary = {
        "rows": len(frame), "columns": len(frame.columns),
        "fraud": int(frame.Class.sum()), "legitimate": int((frame.Class == 0).sum()),
        "fraud_percent": 100 * float(frame.Class.mean()),
        "duplicate_rows": int(frame.duplicated().sum()),
        "missing": frame.isna().sum().to_dict(),
        "dtypes": frame.dtypes.astype(str).to_dict(),
    }
    write_json(tables / "dataset.json", summary)
    frame.head(12).to_csv(tables / "preview.csv", index=False)
    frame.describe().T.to_csv(tables / "summary.csv")
    frame.groupby("Class")[FEATURES].agg(["mean", "median", "std"]).T.to_csv(
        tables / "class_statistics.csv"
    )
    q1, q3 = frame[FEATURES].quantile([0.25, 0.75]).to_numpy()
    iqr = q3 - q1
    outside = (frame[FEATURES] < q1 - 1.5 * iqr) | (frame[FEATURES] > q3 + 1.5 * iqr)
    outside.sum().rename("outside_1_5_iqr").to_csv(tables / "outliers.csv")

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.bar(["Legitimate", "Fraud"], [summary["legitimate"], summary["fraud"]])
    ax.set(yscale="log", ylabel="Transactions (log scale)", title="Observed class imbalance")
    save(fig, directory, "class_distribution")

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    axes[0].hist(np.log1p(frame.Amount.dropna()), bins=60)
    axes[0].set(xlabel="log(1 + Amount)", ylabel="Transactions", title="Transaction amount")
    axes[1].hist(frame.Time.dropna() / 3600, bins=48)
    axes[1].set(xlabel="Hours since first transaction", title="Transaction timing")
    save(fig, directory, "amount_time")

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    plot = frame.assign(LogAmount=np.log1p(frame.Amount))
    for ax, column in zip(axes, ["LogAmount", "V4", "V14"]):
        sns.boxplot(data=plot, x="Class", y=column, showfliers=False, ax=ax)
        ax.set_title(column)
    save(fig, directory, "class_features")

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(frame.corr(), cmap="vlag", center=0, ax=ax)
    ax.set_title("Pearson feature correlations")
    save(fig, directory, "correlation")
    return summary


def embeddings(training, count, seed, directory):
    x = training[FEATURES]
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    full = scaler.fit_transform(imputer.fit_transform(x))
    pca = PCA(svd_solver="full").fit(full)
    fraction = pca.explained_variance_ratio_
    fig, ax = plt.subplots(figsize=(7, 4))
    components = np.arange(1, len(fraction) + 1)
    ax.bar(components, fraction, alpha=0.55, label="Per component")
    ax.plot(components, fraction.cumsum(), label="Cumulative")
    ax.set(xlabel="Component", ylabel="Variance fraction", title="PCA explained variance")
    ax.legend()
    save(fig, directory, "pca_variance")

    # Include sparse positives for visibility while drawing negatives randomly.
    fraud = np.flatnonzero(training.Class.to_numpy() == 1)
    legit = np.flatnonzero(training.Class.to_numpy() == 0)
    rng = np.random.default_rng(seed)
    fraud_count = min(len(fraud), max(1, count // 10))
    chosen = np.r_[rng.choice(legit, min(len(legit), count - fraud_count), replace=False),
                   rng.choice(fraud, fraud_count, replace=False)]
    labels = training.Class.to_numpy()[chosen]
    values = full[chosen]
    coordinates = {
        "pca": pca.transform(values)[:, :2],
        "tsne": TSNE(perplexity=min(30, len(values) - 1), random_state=seed,
                     init="pca", learning_rate="auto").fit_transform(values),
    }
    for name, points in coordinates.items():
        fig, ax = plt.subplots(figsize=(7, 4.2))
        for label, mask in [("Legitimate", labels == 0), ("Fraud", labels == 1)]:
            ax.scatter(*points[mask].T, s=12, alpha=0.58, label=label)
        ax.set(xlabel="Dimension 1", ylabel="Dimension 2",
               title=f"{name.upper()} · training sample with fraud enrichment")
        ax.legend()
        save(fig, directory, name)
    return {"rows": len(chosen), "fraud": int(labels.sum()),
            "pca_first_two_variance": float(fraction[:2].sum())}


def comparison(metrics, curves, directory):
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    table = pd.DataFrame(metrics).T
    for ax, field, title in zip(axes,
        ["average_precision", "training_seconds", "inference_microseconds_per_row"],
        ["Holdout average precision", "Search + final fit (seconds)", "Scoring (microseconds / row)"],
    ):
        table[field].astype(float).sort_values().plot.barh(ax=ax)
        ax.set_title(title)
    save(fig, directory, "comparison")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for name, points in curves.items():
        for ax, key in zip(axes, ["roc", "pr"]):
            ax.plot(points[key]["x"], points[key]["y"], label=name)
    axes[0].set(xlabel="False positive rate", ylabel="True positive rate", title="ROC")
    axes[1].set(xlabel="Recall", ylabel="Precision", title="Precision–recall")
    axes[1].legend(fontsize=8)
    save(fig, directory, "curves")

    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    for ax, (name, row) in zip(axes.flat, metrics.items()):
        sns.heatmap(np.array(row["confusion_matrix"]), annot=True, fmt="d",
                    cmap="Blues", cbar=False, ax=ax)
        ax.set(title=name, xlabel="Predicted", ylabel="Actual")
    for ax in list(axes.flat)[len(metrics):]:
        ax.axis("off")
    save(fig, directory, "confusion_matrices")


def insights(learning, importance, losses, directory):
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for ax, (name, rows) in zip(axes.flat, learning.items()):
        table = pd.DataFrame(rows)
        ax.plot(table.rows, table.train_ap, "o-", label="Training")
        ax.errorbar(table.rows, table.validation_ap, yerr=table.validation_sd,
                    fmt="s-", capsize=3, label="Validation")
        ax.set(title=name, xlabel="Training rows per fold", ylabel="Average precision")
        ax.legend()
    save(fig, directory, "learning_curves")

    fig, axes = plt.subplots(1, 3, figsize=(13, 5))
    for ax, (name, rows) in zip(axes, importance.items()):
        top = sorted(rows, key=lambda row: abs(row["value"]))[-10:]
        ax.barh([row["feature"] for row in top], [row["value"] for row in top])
        ax.set_title(name)
    save(fig, directory, "importance")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(np.arange(1, len(losses) + 1), losses)
    ax.set(xlabel="Epoch", ylabel="Training loss", title="Selected MLP training history")
    save(fig, directory, "mlp_loss")
