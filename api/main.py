"""FastAPI application: health, model metadata and prediction."""
from __future__ import annotations

import time

from fastapi import FastAPI, HTTPException, Request, UploadFile
from PIL import Image

from api import settings
from api.service import ModelUnavailable, OnnxService, build_metadata


def create_app(service=None) -> FastAPI:
    """Build the app. Pass a stub `service` in tests; otherwise ONNX is loaded on first use."""
    app = FastAPI(
        title="Chest X-ray Diagnosis API",
        description="Research/educational demo. Not a diagnostic tool.",
        version="0.1.0",
    )
    app.state.service = service

    def get_service(request: Request):
        svc = request.app.state.service
        if svc is None:
            try:
                svc = request.app.state.service = OnnxService()
            except ModelUnavailable as exc:
                raise HTTPException(status_code=503, detail=str(exc)) from exc
        return svc

    @app.get("/health")
    def health(request: Request) -> dict:
        loaded = request.app.state.service is not None
        return {
            "status": "ok",
            "model_loaded": loaded,
            "model_file_present": loaded or settings.MODEL_ONNX.exists(),
        }

    @app.get("/model/metadata")
    def model_metadata(request: Request) -> dict:
        svc = request.app.state.service
        if svc is not None:
            return svc.metadata
        return build_metadata(settings.MODEL_ONNX, settings.TEST_METRICS)

    @app.post("/predict")
    def predict(request: Request, file: UploadFile) -> dict:
        svc = get_service(request)
        image = Image.open(file.file)
        start = time.perf_counter()
        probs = svc.predict(image)
        latency_ms = (time.perf_counter() - start) * 1000
        scores = {name: float(p) for name, p in zip(svc.classes, probs, strict=True)}
        top = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:5]
        return {
            "probabilities": scores,
            "top": [{"disease": d, "probability": p} for d, p in top],
            "latency_ms": round(latency_ms, 2),
        }

    return app


app = create_app()
