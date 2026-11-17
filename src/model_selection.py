"""Search spaces and training-only model/threshold selection."""
from itertools import product

import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


def candidates(config, min_fold_positives, minority_ratio):
    seed, trees, threads = (config[key] for key in ("seed", "trees", "threads"))
    smote = SMOTE(
        sampling_strategy=min(1.0, max(0.05, minority_ratio * 2)),
        random_state=seed, k_neighbors=min(5, min_fold_positives - 1),
    )
    imbalance = [
        ("passthrough", None), ("passthrough", "balanced"), (smote, None)
    ]
    # In sklearn 1.8, l1_ratio=0 and 1 select pure L2 and L1 respectively.
    logistic = [
        {"model__l1_ratio": [ratio], "model__C": [strength],
         "sample": [sampler], "model__class_weight": [weight]}
        for ratio, strength, (sampler, weight) in product(
            [0, 1], [0.1, 1.0], imbalance
        )
    ]
    return {
        "logistic": (LogisticRegression(solver="liblinear", max_iter=400,
                                        random_state=seed), True, logistic),
        "tree": (DecisionTreeClassifier(random_state=seed), False, {
            "model__max_depth": [3, 8, None], "model__min_samples_split": [2, 20],
            "model__class_weight": [None, "balanced"],
        }),
        "forest": (RandomForestClassifier(n_estimators=trees, random_state=seed,
                                          n_jobs=threads), False, {
            "model__max_depth": [8, None], "model__min_samples_leaf": [1, 5],
            "model__class_weight": [None, "balanced_subsample"],
        }),
        "svm": (SVC(cache_size=256, class_weight="balanced", random_state=seed), True, [
            {"model__kernel": ["linear"], "model__C": [0.1, 1]},
            {"model__kernel": ["rbf"], "model__C": [1, 10],
             "model__gamma": ["scale", 0.01]},
        ]),
        "knn": (KNeighborsClassifier(n_jobs=threads), True, {
            "model__n_neighbors": [5, 15], "scale": [StandardScaler(), "passthrough"],
        }),
        "mlp": (MLPClassifier(random_state=seed, early_stopping=True,
                              n_iter_no_change=8, max_iter=120), True, {
            "model__hidden_layer_sizes": [(16,), (32, 16)],
            "model__alpha": [0.0001, 0.01],
        }),
        "xgboost": (XGBClassifier(n_estimators=trees, tree_method="hist",
                                 n_jobs=threads, random_state=seed,
                                 eval_metric="logloss"), False, {
            "model__max_depth": [3, 6], "model__learning_rate": [0.05, 0.2],
            # A fixed positive weight is itself a hyperparameter, selected in CV.
            "model__scale_pos_weight": [1, 25],
        }),
    }


def fraud_scores(estimator, x):
    if hasattr(estimator, "predict_proba"):
        return estimator.predict_proba(x)[:, 1], "probability"
    return estimator.decision_function(x), "decision_score"


def select_threshold(y, scores) -> float:
    precision, recall, thresholds = precision_recall_curve(y, scores)
    numerator = 2 * precision[:-1] * recall[:-1]
    denominator = precision[:-1] + recall[:-1]
    f1 = np.divide(numerator, denominator, out=np.zeros_like(numerator),
                   where=denominator != 0)
    return float(thresholds[np.flatnonzero(f1 == f1.max())[-1]])
