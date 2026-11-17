import pytest

from src import train
from src.predict import predict
from src.utils import ROOT, read_json


def test_seven_model_experiment(tmp_path, monkeypatch, transactions):
    """Exercise the entire pipeline with a synthetic fixture, isolated from real runs."""
    dataset = tmp_path / "synthetic.csv"
    transactions.to_csv(dataset, index=False)
    models = tmp_path / "models"
    monkeypatch.setattr(train, "MODELS", models)
    monkeypatch.setattr(train, "ROOT", tmp_path)
    config = read_json(ROOT / "configs/fast.json")
    config.update(trees=4, threads=1, embedding_rows=100)
    result = train.run(dataset, config, "synthetic_test")
    assert set(result["models"]) == {"tree", "forest", "logistic", "svm", "knn", "mlp", "xgboost"}
    assert all(0 <= row["average_precision"] <= 1 for row in result["models"].values())
    assert result["best_model"] == max(
        result["models"], key=lambda name: result["models"][name]["selection_average_precision"]
    )
    report = tmp_path / "reports/runs" / result["run_id"]
    assert (report / "figures/tsne.png").is_file()
    assert (report / "figures/learning_curves.png").is_file()
    examples = transactions.drop(columns="Class").iloc[:3]
    assert len(predict(examples, root=models)) == 3
    assert predict(examples, "svm", models).score_kind.tolist() == ["decision_score"] * 3
    assert result["dataset"]["fraud_percent"] == pytest.approx(12.0)
