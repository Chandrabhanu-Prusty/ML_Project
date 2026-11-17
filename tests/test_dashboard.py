import pandas as pd

from app import components


def test_metric_bar_uses_model_index_as_labels(monkeypatch):
    charts = []
    monkeypatch.setattr(components.st, "plotly_chart", lambda chart, **_: charts.append(chart))
    table = pd.DataFrame({"average_precision": [0.8, 0.3]},
                         index=pd.Index(["forest", "tree"], name="model"))
    components.metric_bar(table, "average_precision", "Average precision")
    assert list(charts[0].data[0].y) == ["tree", "forest"]
    assert list(charts[0].data[0].x) == [0.3, 0.8]
