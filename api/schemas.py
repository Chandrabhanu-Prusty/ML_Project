"""Typed API models derived from the canonical transaction columns."""
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, create_model

from src.data_loader import FEATURES
from src.predict import MAX_BATCH

Number = Annotated[float, Field(strict=True, allow_inf_nan=False)]
Nonnegative = Annotated[float, Field(strict=True, ge=0, allow_inf_nan=False)]
Transaction = create_model(
    "Transaction", __config__=ConfigDict(extra="forbid"),
    **{name: (Nonnegative if name in ("Time", "Amount") else Number, ...)
       for name in FEATURES},
)


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    transaction: Transaction
    model: str | None = None


class BatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    transactions: Annotated[list[Transaction], Field(min_length=1, max_length=MAX_BATCH)]
    model: str | None = None


class Prediction(BaseModel):
    prediction: int
    fraud_score: float
    score_kind: str
    model: str
    threshold: float
    run_id: str
