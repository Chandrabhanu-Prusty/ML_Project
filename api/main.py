"""FastAPI application for versioned fraud scoring."""
import pandas as pd
from fastapi import FastAPI, HTTPException

from api.schemas import BatchRequest, Prediction, PredictionRequest
from src.predict import ArtifactsUnavailable, predict, registry
from src.utils import MODELS


def create_app(root=MODELS):
    app = FastAPI(title="Credit Card Fraud Detection", version="1.0.0")

    @app.get("/health")
    def health():
        try:
            _, metadata = registry(root)
            return {"status": "ready", "run_id": metadata["run_id"]}
        except ArtifactsUnavailable as exc:
            return {"status": "awaiting_artifacts", "detail": str(exc)}

    def metadata():
        try:
            return registry(root)[1]
        except ArtifactsUnavailable as exc:
            raise HTTPException(503, str(exc)) from exc

    @app.get("/models")
    def models():
        run = metadata()
        return {"run_id": run["run_id"], "profile": run["profile"],
                "best_model": run["best_model"], "models": list(run["models"])}

    @app.get("/metrics")
    def metrics():
        return metadata()

    def score(transactions, model):
        try:
            frame = pd.DataFrame([row.model_dump() for row in transactions])
            return predict(frame, model, root).to_dict("records")
        except ArtifactsUnavailable as exc:
            raise HTTPException(503, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.post("/predict", response_model=Prediction)
    def predict_one(request: PredictionRequest):
        return score([request.transaction], request.model)[0]

    @app.post("/predict-batch", response_model=list[Prediction])
    def predict_batch(request: BatchRequest):
        return score(request.transactions, request.model)

    return app


app = create_app()
