"""Evaluate final models once and save bounded-size curve data for the UI."""
from time import perf_counter

import numpy as np
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix, f1_score,
    precision_recall_curve, precision_score, recall_score, roc_auc_score, roc_curve,
)

from src.model_selection import fraud_scores


def evaluate(model, x, y, threshold):
    start = perf_counter()
    score, kind = fraud_scores(model, x)
    duration = perf_counter() - start
    prediction = score >= threshold
    matrix = confusion_matrix(y, prediction, labels=[0, 1])
    tn, fp, fn, tp = (int(value) for value in matrix.ravel())
    fpr, tpr, _ = roc_curve(y, score)
    precision, recall, _ = precision_recall_curve(y, score)
    metrics = {
        "average_precision": average_precision_score(y, score),
        "roc_auc": roc_auc_score(y, score),
        "accuracy": accuracy_score(y, prediction),
        "precision": precision_score(y, prediction, zero_division=0),
        "recall": recall_score(y, prediction, zero_division=0),
        "f1": f1_score(y, prediction, zero_division=0),
        "false_positive_rate": fp / (fp + tn),
        "false_negative_rate": fn / (fn + tp),
        "false_positives": fp, "false_negatives": fn,
        "inference_seconds": duration,
        "inference_microseconds_per_row": duration * 1e6 / len(y),
        "confusion_matrix": matrix.tolist(),
        "score_kind": kind, "threshold": threshold,
    }
    curves = {"roc": curve(fpr, tpr), "pr": curve(recall, precision)}
    return metrics, curves


def curve(x, y):
    indices = np.linspace(0, len(x) - 1, min(len(x), 3000), dtype=int)
    return {"x": x[indices].tolist(), "y": y[indices].tolist()}
