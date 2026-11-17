"""Synthetic transactions for tests only."""
import numpy as np
import pandas as pd
import pytest

from src.data_loader import FEATURES


@pytest.fixture
def transactions():
    rng = np.random.default_rng(42)
    x = pd.DataFrame(rng.normal(size=(300, len(FEATURES))), columns=FEATURES)
    x[["Time", "Amount"]] = x[["Time", "Amount"]].abs()
    y = np.zeros(len(x), dtype=int)
    y[:36] = 1
    x.loc[y == 1, "V1"] += 3
    return x.assign(Class=y)
