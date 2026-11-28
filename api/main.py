"""FastAPI application: health, models, prediction, Grad-CAM and model comparison."""
from __future__ import annotations

import base64
import io
import time

from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from api import settings
from api.images import read_image
from api.service import ModelHub, ModelUnavailable, UnknownModel, build_metadata, default_entry


def create_app(service=None) -> FastAPI:
    """Build the app. Pass a stub `service` in tests; otherwise models load on first use."""
    app = FastAPI(
        title="Chest X-ray Diagnosis API",
        description="Research/educational demo. Not a diagnostic tool.",
        version="0.2.0",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )
    app.state.service = service

    def get_service(request: Request):
        svc = request.app.state.service
        if svc is None:
            try:
                svc = request.app.state.service = ModelHub()
            except ModelUnavailable as exc:
                raise HTTPException(status_code=503, detail=str(exc)) from exc
        return svc

    def check_model(svc, model: str | None) -> None:
        if model is not None and model not in svc.model_ids:
            available = ", ".join(svc.model_ids)
            raise HTTPException(400, detail=f"Unknown model: {model}. Available: {available}")

    @app.get("/health")
    def health(request: Request) -> dict:
        loaded = request.app.state.service is not None
        present = loaded or settings.MODEL_ONNX.exists() or settings.REGISTRY.exists()
        return {"status": "ok", "model_loaded": loaded, "model_file_present": present}

    @app.get("/model/metadata")
    def model_metadata(request: Request) -> dict:
        svc = request.app.state.service
        if svc is not None:
            return svc.metadata
        return build_metadata(default_entry())

    @app.get("/models")
    def list_models(request: Request) -> list[dict]:
        return get_service(request).models_info()

    @app.post("/predict")
    def predict(request: Request, file: UploadFile, model: str | None = None) -> dict:
        svc = get_service(request)
        check_model(svc, model)
        image = read_image(file)
        start = time.perf_counter()
        probs = _call(svc.predict, image, model=model)
        return _prediction_payload(svc, probs, start) | {"model": model or svc.default_id}

    @app.post("/predict/visualize")
    def predict_visualize(
        request: Request, file: UploadFile, disease: str, model: str | None = None
    ) -> dict:
        svc = get_service(request)
        check_disease(svc, disease)
        check_model(svc, model)
        image = read_image(file)
        start = time.perf_counter()
        probs = _call(svc.predict, image, model=model)
        heat = _call(svc.visualize, image, disease, model=model)
        payload = _prediction_payload(svc, probs, start)
        return payload | {"model": model or svc.default_id, "disease": disease,
                          "overlay_png_base64": _png_base64(heat)}

    @app.post("/predict/compare")
    def predict_compare(request: Request, file: UploadFile, disease: str | None = None) -> dict:
        """Every model (and the ensemble) on the same image; heatmaps if a disease is given."""
        svc = get_service(request)
        if disease is not None:
            check_disease(svc, disease)
        image = read_image(file)
        results = []
        for model_id in svc.model_ids:
            start = time.perf_counter()
            probs = _call(svc.predict, image, model=model_id)
            item = _prediction_payload(svc, probs, start) | {"model": model_id}
            if disease is not None:
                item["overlay_png_base64"] = _png_base64(
                    _call(svc.visualize, image, disease, model=model_id)
                )
                item["latency_ms"] = round((time.perf_counter() - start) * 1000, 2)
            results.append(item)
        return {"disease": disease, "results": results}

    def check_disease(svc, disease: str) -> None:
        if disease not in svc.classes:
            raise HTTPException(status_code=400, detail=f"Unknown disease: {disease}")

    return app


def _call(fn, *args, model=None):
    """Call a service method, passing `model` only when one is requested (stub-friendly)."""
    try:
        return fn(*args, model=model) if model is not None else fn(*args)
    except UnknownModel as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ModelUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


def _png_base64(rgb) -> str:
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


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
