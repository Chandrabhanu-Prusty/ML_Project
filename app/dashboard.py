"""Streamlit dashboard for comparing experiments and scoring transactions."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# Streamlit executes this file directly, with app/ on the import path.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.components import curve_chart, kpis, metric_bar, show_figures
from src.data_loader import FEATURES, SOURCE
from src.predict import ArtifactsUnavailable, MAX_BATCH, predict, predict_csv, registry
from src.utils import ROOT, read_json

PAGES = ("Overview", "Exploratory Data Analysis", "Model Comparison", "Fraud Prediction",
         "Dimensionality Reduction", "Model Insights", "About Project")
FIELDS = {
    "average_precision": "Average precision", "roc_auc": "ROC AUC",
    "precision": "Precision", "recall": "Recall", "f1": "F1",
    "accuracy": "Accuracy", "false_positive_rate": "False positive rate",
    "false_negative_rate": "False negative rate", "false_positives": "False positives",
    "false_negatives": "Missed fraud cases",
    "training_seconds": "Search and fit (s)",
    "inference_microseconds_per_row": "Scoring (μs / row)",
}


def overview(metadata, report):
    st.write("Compare seven models on the ULB / Worldline transactions and explore how "
             "their operating thresholds affect fraud detection and review workload.")
    summary = metadata["dataset"]
    kpis([("Transactions", f'{summary["rows"]:,}'),
          ("Fraud", f'{summary["fraud"]:,}'),
          ("Legitimate", f'{summary["legitimate"]:,}'),
          ("Fraud rate", f'{summary["fraud_percent"]:.3f}%')])
    selected = metadata["best_model"]
    metrics = metadata["models"][selected]
    st.subheader(f"Selected model · {selected}")
    st.write("Selected by average precision on training-side validation. The scores below come from the holdout.")
    kpis([("Average precision", f'{metrics["average_precision"]:.3f}'),
          ("Recall", f'{metrics["recall"]:.3f}'),
          ("False positives", str(metrics["false_positives"])),
          ("Holdout fraud cases", str(metadata["test_fraud"]))])
    show_figures(report, [("class_distribution", "Observed label counts before modeling exclusions")])


def eda_page(metadata, report):
    summary = metadata["dataset"]
    st.caption(f'{summary["columns"]} columns · {summary["duplicate_rows"]:,} exact duplicate rows')
    st.dataframe(pd.read_csv(report / "preview.csv"), hide_index=True)
    tabs = st.tabs(["Distributions", "Correlations", "Data quality"])
    with tabs[0]:
        show_figures(report, [("class_distribution", "Class frequencies"),
                              ("amount_time", "Amount and elapsed time"),
                              ("class_features", "Selected features by observed class")])
    with tabs[1]:
        show_figures(report, [("correlation", "Pearson correlations, including the observed label")])
    with tabs[2]:
        st.dataframe(pd.DataFrame({"Type": summary["dtypes"], "Missing": summary["missing"]}))
        for name in ("summary", "class_statistics", "outliers"):
            with st.expander(name.replace("_", " ").capitalize()):
                st.dataframe(pd.read_csv(report / f"{name}.csv"))


def compare_page(metadata, report):
    table = pd.DataFrame.from_dict(metadata["models"], orient="index").rename_axis("model")
    names = st.multiselect("Models", table.index.tolist(), default=table.index.tolist())
    if not names:
        st.info("Select at least one model.")
        return
    field = st.selectbox("Holdout metric", list(FIELDS), format_func=FIELDS.get)
    chosen = table.loc[names]
    metric_bar(chosen, field, FIELDS[field])
    st.dataframe(chosen[[*FIELDS, "cv_average_precision", "cv_sd", "fit_rows",
                         "selection_average_precision"]].rename(columns=FIELDS))
    st.caption("SVM and KNN use capped training subsets. Scoring time measures one complete holdout batch.")
    curves = read_json(report / "curves.json")
    for tab, key in zip(st.tabs(["Precision–recall", "ROC", "Confusion matrix"]),
                        ["pr", "roc", "matrix"]):
        with tab:
            if key != "matrix":
                curve_chart(curves, names, key)
            else:
                name = st.selectbox("Model for matrix", names)
                chart = px.imshow(np.array(table.loc[name, "confusion_matrix"]), text_auto=True,
                                  x=["Legitimate", "Fraud"], y=["Legitimate", "Fraud"],
                                  labels={"x": "Predicted", "y": "Actual"},
                                  color_continuous_scale="Teal")
                st.plotly_chart(chart, width="stretch")


def prediction_page(metadata, report):
    chosen = st.selectbox("Model", list(metadata["models"]),
                          index=list(metadata["models"]).index(metadata["best_model"]))
    st.caption("V1–V28 are anonymized PCA features. Weighted or resampled model scores may be poorly calibrated.")
    manual, batch = st.tabs(["Single transaction", "CSV batch"])
    with manual:
        examples = pd.read_csv(report / "held_out_examples.csv")
        option = st.selectbox("Starting values", ["Blank", *[
            f"Holdout row {i + 1} · {'fraud' if label else 'legitimate'}"
            for i, label in enumerate(examples.Class)
        ]])
        index = 0 if option == "Blank" else int(option.split()[2])
        start = dict.fromkeys(FEATURES, 0.0) if not index else examples.iloc[index - 1].to_dict()
        with st.form(f"transaction-{index}"):
            columns = st.columns(3)
            row = {}
            for i, name in enumerate(FEATURES):
                default = start[name] if pd.notna(start[name]) else 0.0
                row[name] = columns[i % 3].number_input(name, value=float(default), format="%.8f")
            submitted = st.form_submit_button("Score transaction", type="primary")
        if submitted:
            try:
                result = predict(pd.DataFrame([row]), chosen).iloc[0]
                st.success("Flagged as fraud" if result.prediction else "Classified as legitimate")
                kpis([("Fraud score", f"{result.fraud_score:.4f}"),
                      ("Threshold", f"{result.threshold:.4f}"), ("Score type", result.score_kind)])
            except (ArtifactsUnavailable, ValueError) as exc:
                st.error(str(exc))
    with batch:
        st.caption(f"Upload up to {MAX_BATCH:,} rows with Time, V1–V28 and Amount.")
        st.download_button("Download example CSV", examples[FEATURES].to_csv(index=False),
                           file_name="held_out_examples.csv", mime="text/csv")
        upload = st.file_uploader("Transaction CSV", type=["csv"])
        if st.button("Score CSV", disabled=upload is None, type="primary"):
            try:
                st.session_state["batch"] = (upload.file_id, chosen, predict_csv(upload, chosen))
            except (ArtifactsUnavailable, ValueError) as exc:
                st.session_state.pop("batch", None)
                st.error(str(exc))
        saved = st.session_state.get("batch")
        if upload is not None and saved and saved[:2] == (upload.file_id, chosen):
            st.dataframe(saved[2], hide_index=True)
            st.download_button("Download scored CSV", saved[2].to_csv(index=False),
                               file_name="fraud_predictions.csv", mime="text/csv")


def embedding_page(metadata, report):
    sample = metadata["embedding"]
    st.write(f'Training visualization sample of {sample["rows"]:,} rows, including '
             f'{sample["fraud"]:,} fraud examples. Sampling enriches fraud to make it visible.')
    st.caption(f'The first two PCA axes explain {sample["pca_first_two_variance"]:.1%} of training variance.')
    show_figures(report, [("pca_variance", "PCA variance fractions"),
                          ("pca", "Projection onto the first two PCA components"),
                          ("tsne", "t-SNE local structure. Visual separation does not measure classifier performance.")])


def insights_page(metadata, report):
    st.write(f'Learning diagnostics use a stratified training subset of '
             f'{metadata["diagnostic_rows"]:,} rows. A persistent training–validation '
             'gap suggests overfitting. Low scores on both can indicate underfitting.')
    show_figures(report, [("learning_curves", "Training and validation average precision"),
                          ("importance", "Largest absolute coefficients and impurity importance"),
                          ("mlp_loss", "Training loss across epochs")])
    model = st.selectbox("Cross-validation trials", list(metadata["models"]))
    trials = pd.read_csv(report / f"cv_{model}.csv")
    columns = [column for column in trials if column.startswith("param_")]
    st.dataframe(trials[[*columns, "mean_train_score", "mean_test_score", "std_test_score",
                         "mean_fit_time"]].sort_values("mean_test_score", ascending=False))
    st.caption("Logistic l1_ratio 1 means L1 and 0 means L2. Smaller C increases regularization.")


def about():
    st.markdown(f"""
The project compares seven classifiers for rare fraud detection using the
[ULB / Worldline Credit Card Fraud Detection dataset]({SOURCE}).

**Models** · Logistic regression, decision tree, random forest, SVM, KNN, MLP and XGBoost.

**Method** · Stratified holdout, fold-local preprocessing, average precision for selection
and validation-only threshold tuning.

**Stack** · Python, scikit-learn, imbalanced-learn, XGBoost, FastAPI and Streamlit.

**Limitations** · The anonymized dataset covers a short window, has no cardholder identifier,
and cannot establish performance on future traffic. Scores are experimental.

**Future work** · Temporal validation, calibration, cost-based thresholds and drift monitoring.
""")


def main():
    st.set_page_config(page_title="Credit Card Fraud Detection", page_icon="📊", layout="wide")
    st.sidebar.title("Fraud Detection")
    page = st.sidebar.radio("Explore", PAGES)
    st.title("Comparative Analysis of ML Models for Credit Card Fraud Detection"
             if page == "Overview" else page)
    if page == "About Project":
        about()
        return
    try:
        _, metadata = registry()
    except ArtifactsUnavailable as exc:
        st.info(str(exc))
        st.code("python main.py --mode train --fast", language="bash")
        st.markdown(f"Place the [ULB dataset]({SOURCE}) in `data/raw/creditcard.csv`.")
        return
    report = ROOT / "reports/runs" / metadata["run_id"]
    if not (report / "metadata.json").is_file():
        st.error("The report files for this model run are missing. Re-run training.")
        return
    st.sidebar.caption(f'{metadata["profile"].capitalize()} experiment · {metadata["run_id"]}')
    if metadata["profile"] == "fast":
        st.warning("Development run. Fewer training rows and CV folds reduce reliability.")
    views = dict(zip(PAGES, (overview, eda_page, compare_page, prediction_page,
                             embedding_page, insights_page)))
    views[page](metadata, report)


if __name__ == "__main__":
    main()
