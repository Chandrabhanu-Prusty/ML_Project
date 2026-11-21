# Comparative Analysis of ML Models for Credit Card Fraud Detection

This academic project compares seven supervised classifiers for rare fraud detection in the ULB / Worldline Credit Card Fraud Detection dataset. It covers data profiling, imbalance controls, model tuning, a held-out evaluation, and single and batch scoring.

The [academic report](reports/project_report.md) explains the experiment. The [measured results](reports/results.md) give the recorded metrics and figures. The [syllabus mapping](syllabus_mapping.md) and [viva guide](viva_questions.md) accompany the five notebooks.

## Install

Python 3.14 was used for verification. Create a virtual environment in the project root. The commands below use PowerShell.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

On macOS or Linux, use `.venv/bin/python` in place of `.\.venv\Scripts\python.exe`.

## Dataset

Download [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and extract `creditcard.csv` into `data/raw/`. The dataset attributes the transactions to the Worldline / Université Libre de Bruxelles research collaboration. The schema uses elapsed seconds (`Time`), 28 anonymized PCA components (`V1`–`V28`), `Amount`, and a binary `Class` (1 denotes fraud).

If you have configured the Kaggle CLI, run:

```powershell
kaggle datasets download -d mlg-ulb/creditcardfraud -p data/raw --unzip
```

The loader reports a missing file with the source and expected path. The Git ignore rules exclude the dataset, raw sample exports and serialized models.

## Train and run

Run each service in its own terminal after training.

```powershell
.\.venv\Scripts\python.exe main.py --mode train --fast
.\.venv\Scripts\python.exe main.py --mode train
.\.venv\Scripts\python.exe -m uvicorn api.main:app --host 127.0.0.1 --port 8000
.\.venv\Scripts\python.exe -m streamlit run app/dashboard.py --server.address 127.0.0.1
.\.venv\Scripts\python.exe -m pytest -q
```

- Open the [Streamlit dashboard](http://127.0.0.1:8501). Use **Fraud Prediction** to load a holdout example, edit a transaction, or download and re-upload a batch CSV.
- Open [Swagger UI](http://127.0.0.1:8000/docs) for the JSON API. `GET /health` reports registry readiness. `GET /models` lists the available model IDs. `GET /metrics` exposes the run metadata and holdout results.
- For `POST /predict`, send `{"transaction": {...}, "model": "logistic"}` with all 30 numeric features in the object. For `POST /predict-batch`, send `{"transactions": [{...}, {...}]}`. Omit `model` to use validation's selected model. Both endpoints return the binary prediction, raw fraud score, score type, threshold and run ID.
- Batch CSVs accept exactly the 30 feature columns in any order, with up to 10,000 rows. Use the downloadable example as a schema reference. Scores labeled probability come from the estimator and may be poorly calibrated. SVM returns a signed margin.
- Use `--data path/to/creditcard.csv` for a different path or `--config path/to/profile.json` to override the run config. Optional MLflow logging requires `python -m pip install mlflow`, then `python main.py --mode train --mlflow`. The JSON, CSV, and figure log works without MLflow.

## Architecture and experiment

![Architecture Diagram](Architecture%20diagram.png)

```text
validated CSV
  ├─ observed-data EDA
  └─ deduplicate → stratified train / holdout
       └─ training pool → fit partition + threshold validation
            ├─ stratified CV → fit imputer, scaler and optional SMOTE per fold
            ├─ select parameters → refit on the fit partition
            └─ validation AP selects model → validation F1 sets thresholds
       └─ evaluate each chosen configuration on the common holdout
            → versioned Joblib pipelines, metrics and figures
            → Streamlit and FastAPI use the same scoring code
```

The default profile uses five folds. The fast profile draws a stratified 14,000-row development pool and uses three folds. Both evaluate against the same held-out split. SVM and KNN have explicit fitting caps to bound kernel training and neighbor search. Fast scores demonstrate the workflow and are sensitive to the small fraud count in its fitting and threshold partitions.

The logistic grid compares no balancing, class weights and fold-local SMOTE across L1/L2 and two regularization strengths. Trees and XGBoost compare complexity and weighting. SVM compares linear and RBF kernels. KNN compares both neighborhood size and scaling. MLP compares architectures and regularization with early stopping. Average precision (AP) summarizes score ranking over the precision–recall curve. Accuracy alone is dominated by legitimate transactions. The report includes threshold-based errors and the false-positive workload.

A completed run publishes the selected model ID and all seven pipelines. The `models/latest.json` pointer switches only after training and report generation complete. Joblib inputs are trusted local training artifacts. Keep the registry directory writable only by trusted operators.

To update the Markdown results and presentation figures from the current run, execute `python -m src.publish_report` in the project environment.

## Layout

```text
app/                 Streamlit dashboard and components
api/                 FastAPI routing and Pydantic schemas
configs/             Development and full experiment settings
data/raw/            Local source dataset
data/processed/      Reserved for derived local data
models/              Run-specific pipelines and latest-run pointer
notebooks/           Five guided study notebooks
reports/             Academic report, measured results and figures
reports/runs/        Local CSV, JSON, figures and held-out examples per run
src/                 Loading, preprocessing, training, evaluation and inference
tests/               Synthetic unit and integration fixtures
main.py              Training CLI
```

To run the notebooks, install JupyterLab with `python -m pip install jupyterlab`, train the project and run `python -m jupyter lab` from the project root. The notebooks use the same source modules and read the current run outputs.

## Limitations and future work

The short observation window and anonymized features prevent temporal or cardholder-level generalization claims. Exact duplicate inputs are excluded before modeling and outliers are retained. Reported latency measures offline batch scoring. Temporal validation, probability calibration, asymmetric investigation costs, confidence intervals, and drift monitoring would improve a follow-on experiment.
