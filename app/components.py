"""Reusable dashboard presentation helpers."""
import pandas as pd
import plotly.express as px
import streamlit as st


def kpis(items):
    for column, (label, value) in zip(st.columns(len(items)), items):
        column.metric(label, value, border=True)


def show_figures(report, names):
    for name, caption in names:
        path = report / "figures" / f"{name}.png"
        if path.is_file():
            st.image(str(path), caption=caption, width="stretch")


def metric_bar(table, field, label):
    figure = px.bar(table.sort_values(field).reset_index(), x=field, y=table.index.name,
                    orientation="h", labels={field: label, "model": "Model"},
                    color_discrete_sequence=["#137b79"],
                    template="plotly_white")
    st.plotly_chart(figure, width="stretch")


def curve_chart(curves, models, key):
    rows = [pd.DataFrame(curves[name][key]).assign(Model=name) for name in models]
    x_label, y_label = ("False positive rate", "True positive rate") if key == "roc" else (
        "Recall", "Precision"
    )
    frame = pd.concat(rows, ignore_index=True).rename(columns={"x": x_label, "y": y_label})
    figure = px.line(frame, x=x_label, y=y_label, color="Model",
                     template="plotly_white")
    st.plotly_chart(figure, width="stretch")
