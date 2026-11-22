"""FastAPI application: health, model metadata and (in later commits) prediction."""
from __future__ import annotations

from fastapi import FastAPI

from api import settings
from api.service import build_metadata


def create_app() -> FastAPI:
    app = FastAPI(
        title="Chest X-ray Diagnosis API",
        description="Research/educational demo. Not a diagnostic tool.",
        version="0.1.0",
    )

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "model_file_present": settings.MODEL_ONNX.exists()}

    @app.get("/model/metadata")
    def model_metadata() -> dict:
        return build_metadata(settings.MODEL_ONNX, settings.TEST_METRICS)

    return app


app = create_app()
