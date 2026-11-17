import numpy as np
import pandas as pd
import pytest
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression

from src.data_loader import FEATURES, features, load_dataset, validate_dataset
from src.preprocessing import pipeline, split_data


def test_load_and_order(transactions, tmp_path):
    path = tmp_path / "creditcard.csv"
    transactions[transactions.columns[::-1]].to_csv(path, index=False)
    result = load_dataset(path)
    assert result.columns.tolist() == [*FEATURES, "Class"]
    assert len(result) == 300


@pytest.mark.parametrize("corruption", ["missing", "target", "nonnumeric", "infinite", "all_missing"])
def test_invalid_schema(transactions, corruption):
    if corruption == "missing":
        transactions = transactions.drop(columns="V1")
    if corruption == "target":
        transactions.loc[0, "Class"] = 2
    if corruption == "nonnumeric":
        transactions["Amount"] = "bad"
    if corruption == "infinite":
        transactions.loc[0, "V1"] = np.inf
    if corruption == "all_missing":
        transactions["V1"] = np.nan
    with pytest.raises(ValueError):
        validate_dataset(transactions)


def test_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="Download"):
        load_dataset(tmp_path / "creditcard.csv")


def test_split_deduplicates_and_preserves_prevalence(transactions):
    source = pd.concat([transactions, transactions.iloc[[0]]], ignore_index=True)
    train, test, exclusions = split_data(source, 0.2, 42)
    assert len(train) == 240 and len(test) == 60
    assert set(train.index).isdisjoint(test.index)
    assert exclusions == {"exact_duplicates_removed": 1, "conflicting_rows_removed": 0}
    assert abs(train.Class.mean() - test.Class.mean()) < 0.01
    pd.testing.assert_frame_equal(train, split_data(source, 0.2, 42)[0])


def test_conflicting_feature_rows_stay_out(transactions):
    conflict = transactions.iloc[[0]].assign(Class=0)
    train, test, exclusions = split_data(
        pd.concat([transactions, conflict], ignore_index=True), 0.2, 42
    )
    assert exclusions["conflicting_rows_removed"] == 2
    assert 0 not in train.index and 0 not in test.index


def test_scaler_and_imputer_fit_only_on_training_rows(transactions):
    transactions.loc[11, "V1"] = np.nan
    valid = validate_dataset(transactions)
    train, test, _ = split_data(valid, 0.2, 42)
    test = test.assign(V2=1_000_000)
    model = pipeline(LogisticRegression()).fit(train[FEATURES], train.Class)
    imputed = model.named_steps["imputer"].transform(train[FEATURES])
    np.testing.assert_allclose(model.named_steps["scale"].mean_, imputed.mean(axis=0))
    assert len(model.predict(test[FEATURES])) == len(test)
    assert model.named_steps["imputer"].statistics_[1] == pytest.approx(train.V1.median())
    with pytest.raises(ValueError):
        features(valid.loc[[11], FEATURES])


def test_smote_runs_only_at_fit(transactions):
    train, test, _ = split_data(transactions, 0.2, 42)
    estimator = pipeline(LogisticRegression()).set_params(sample=SMOTE(random_state=42))
    estimator.fit(train[FEATURES], train.Class)
    sampler = estimator.named_steps["sample"]
    assert sampler.sampling_strategy_[1] > 0
    assert estimator.predict_proba(test[FEATURES]).shape == (len(test), 2)
