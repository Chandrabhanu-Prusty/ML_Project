from io import StringIO

import joblib
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression

from api.main import create_app
from src.data_loader import FEATURES
from src.predict import ArtifactsUnavailable, predict, predict_csv
from src.preprocessing import pipeline, split_data
from src.utils import write_json

RUN = "20260101T000000000000Z"


@pytest.fixture
def serving(tmp_path, transactions):
    train, test, _ = split_data(transactions, 0.2, 42)
    estimator = pipeline(LogisticRegression()).fit(train[FEATURES], train.Class)
    directory = tmp_path / RUN
    directory.mkdir()
    joblib.dump(estimator, directory / "logistic.joblib")
    write_json(tmp_path / "latest.json", {"run_id": RUN})
    write_json(directory / "metadata.json", {
        "run_id": RUN, "features": FEATURES, "profile": "test",
        "best_model": "logistic", "models": {"logistic": {"threshold": 0.4}},
        "packages": {},
    })
    return tmp_path, test[FEATURES], estimator


def test_prediction_matches_persisted_pipeline(serving):
    root, x, estimator = serving
    result = predict(x[x.columns[::-1]], root=root)
    assert result.fraud_score.tolist() == pytest.approx(estimator.predict_proba(x)[:, 1])
    assert result.prediction.tolist() == (result.fraud_score >= 0.4).astype(int).tolist()
    assert result.score_kind.unique().tolist() == ["probability"]


def test_batch_csv(serving):
    root, x, _ = serving
    result = predict_csv(StringIO(x.to_csv(index=False)), root=root)
    assert list(result.columns[:30]) == FEATURES
    assert len(result) == len(x)
    pd.testing.assert_series_equal(result.fraud_score,
                                   predict(x, root=root).fraud_score.reset_index(drop=True))


def test_health_missing_artifacts(tmp_path, transactions):
    client = TestClient(create_app(tmp_path))
    assert client.get("/health").json()["status"] == "awaiting_artifacts"
    assert client.get("/models").status_code == 503
    request = {"transaction": transactions.loc[0, FEATURES].to_dict()}
    assert client.post("/predict", json=request).status_code == 503


def test_api_endpoints(serving):
    root, x, _ = serving
    client = TestClient(create_app(root))
    rows = x.head(2).to_dict("records")
    assert client.get("/health").json() == {"status": "ready", "run_id": RUN}
    assert client.get("/models").json()["best_model"] == "logistic"
    assert client.get("/metrics").status_code == 200
    assert client.get("/docs").status_code == 200
    response = client.post("/predict", json={"transaction": rows[0]})
    assert response.status_code == 200
    assert response.json()["run_id"] == RUN
    assert response.json()["threshold"] == 0.4
    batch = client.post("/predict-batch", json={"transactions": rows})
    assert batch.status_code == 200
    assert batch.json()[0] == response.json()
    assert len(batch.json()) == 2


@pytest.mark.parametrize("modification", [{"V1": "1.0"}, {"Amount": -1}, {"Class": 1}, {"Time": None}])
def test_api_invalid_input(serving, modification):
    root, x, _ = serving
    client = TestClient(create_app(root))
    row = x.iloc[0].to_dict() | modification
    assert client.post("/predict", json={"transaction": row}).status_code == 422
    assert client.post("/predict-batch", json={"transactions": []}).status_code == 422


def test_package_mismatch(serving):
    root, x, _ = serving
    from src.utils import read_json

    path = root / RUN / "metadata.json"
    metadata = read_json(path)
    metadata["packages"]["scikit-learn"] = "0"
    write_json(path, metadata)
    with pytest.raises(ArtifactsUnavailable, match="scikit-learn"):
        predict(x, root=root)
