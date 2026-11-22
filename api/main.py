"""FastAPI application: health, model metadata and prediction."""
from __future__ import annotations

import base64
import io
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
        return _prediction_payload(svc, probs, start)

    @app.post("/predict/visualize")
    def predict_visualize(request: Request, file: UploadFile, disease: str) -> dict:
        svc = get_service(request)
        if disease not in svc.classes:
            raise HTTPException(status_code=400, detail=f"Unknown disease: {disease}")
        image = Image.open(file.file)
        start = time.perf_counter()
        probs = svc.predict(image)
        try:
            overlay = svc.visualize(image, disease)
        except ModelUnavailable as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        payload = _prediction_payload(svc, probs, start)
        buf = io.BytesIO()
        Image.fromarray(overlay).save(buf, format="PNG")
        payload["disease"] = disease
        payload["overlay_png_base64"] = base64.b64encode(buf.getvalue()).decode("ascii")
        return payload

    return app


def _prediction_payload(svc, probs, start: float) -> dict:
    """Shared response body: all probabilities, top 5 and elapsed time since `start`."""
    scores = {name: float(p) for name, p in zip(svc.classes, probs, strict=True)}
    top = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:5]
    return {
        "probabilities": scores,
        "top": [{"disease": d, "probability": p} for d, p in top],
        "latency_ms": round((time.perf_counter() - start) * 1000, 2),
    }


app = create_app()
