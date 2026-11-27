import base64
import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from api import settings
from api.main import create_app
from api.preprocess import preprocess
from api.service import ModelUnavailable
from src.config import DISEASES


class StubService:
    """Stands in for ModelHub so the API can be tested without model weights."""

    classes = DISEASES
    metadata = {"model_name": "stub", "classes": DISEASES, "input_size": 224}
    model_ids = ["stub_a", "stub_b", "ensemble"]
    default_id = "stub_a"

    def __init__(self, visualize_error: Exception | None = None):
        self.visualize_error = visualize_error
        self.calls: list[str | None] = []

    def models_info(self):
        return [{"model_id": m, "default": m == self.default_id} for m in self.model_ids]

    def predict(self, image, model=None):
        self.calls.append(model)
        probs = np.linspace(0.05, 0.95, len(DISEASES))
        probs[DISEASES.index("Cardiomegaly")] = 0.99  # make the top class predictable
        if model == "stub_b":
            probs[DISEASES.index("Cardiomegaly")] = 0.10
        return probs

    def visualize(self, image, disease, model=None):
        if self.visualize_error:
            raise self.visualize_error
        return np.full((224, 224, 3), 128, dtype=np.uint8)


def png_bytes(size=(64, 64)) -> bytes:
    rng = np.random.default_rng(0)
    arr = (rng.random(size) * 255).astype("uint8")
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, "PNG")
    return buf.getvalue()


@pytest.fixture
def client():
    return TestClient(create_app(StubService()))


def upload(client, data: bytes, path="/predict", **params):
    files = {"file": ("x.png", io.BytesIO(data), "image/png")}
    return client.post(path, params=params, files=files)


def test_health_reports_model_loaded(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["model_loaded"] is True


def test_metadata_comes_from_service(client):
    assert client.get("/model/metadata").json()["model_name"] == "stub"


def test_predict_returns_probabilities_top_and_latency(client):
    r = upload(client, png_bytes())
    assert r.status_code == 200
    body = r.json()
    assert list(body["probabilities"]) == DISEASES
    assert len(body["top"]) == 5
    assert body["top"][0]["disease"] == "Cardiomegaly"
    assert body["latency_ms"] >= 0


@pytest.mark.parametrize("data", [b"not an image", b""])
def test_predict_rejects_invalid_uploads(client, data):
    assert upload(client, data).status_code == 400


def test_predict_rejects_oversized_upload(client, monkeypatch):
    data = png_bytes((200, 200))
    monkeypatch.setattr(settings, "MAX_UPLOAD_BYTES", len(data) - 1)
    assert upload(client, data).status_code == 413


def test_visualize_returns_decodable_overlay(client):
    r = upload(client, png_bytes(), "/predict/visualize", disease="Mass")
    assert r.status_code == 200
    body = r.json()
    assert body["disease"] == "Mass" and "probabilities" in body
    overlay = Image.open(io.BytesIO(base64.b64decode(body["overlay_png_base64"])))
    assert overlay.size == (224, 224)


def test_visualize_rejects_unknown_disease(client):
    assert upload(client, png_bytes(), "/predict/visualize", disease="Nope").status_code == 400


def test_visualize_503_when_pytorch_weights_missing():
    app = create_app(StubService(visualize_error=ModelUnavailable("no weights")))
    r = upload(TestClient(app), png_bytes(), "/predict/visualize", disease="Mass")
    assert r.status_code == 503


def test_predict_503_when_no_model_file(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "MODEL_ONNX", tmp_path / "missing.onnx")
    client = TestClient(create_app())  # no stub: tries to load the real service
    assert upload(client, png_bytes()).status_code == 503
    assert client.get("/health").json()["model_file_present"] is False


def test_cors_allows_configured_origin_only(client):
    headers = {"Access-Control-Request-Method": "POST"}
    ok = client.options("/predict", headers={**headers, "Origin": "http://localhost:5173"})
    assert ok.headers.get("access-control-allow-origin") == "http://localhost:5173"
    bad = client.options("/predict", headers={**headers, "Origin": "http://evil.example"})
    assert "access-control-allow-origin" not in bad.headers


def test_preprocess_matches_torchvision_eval_transforms():
    pytest.importorskip("torchvision")
    from src.data.transforms import eval_transforms

    rng = np.random.default_rng(0)
    img = Image.fromarray((rng.random((600, 520)) * 255).astype("uint8"))  # grayscale, non-square
    expected = eval_transforms(224)(img.convert("RGB")).unsqueeze(0).numpy()
    np.testing.assert_allclose(preprocess(img), expected, atol=1e-5)


def test_models_lists_every_model_and_the_ensemble(client):
    ids = [m["model_id"] for m in client.get("/models").json()]
    assert ids == ["stub_a", "stub_b", "ensemble"]


def test_predict_routes_to_the_requested_model():
    svc = StubService()
    client = TestClient(create_app(svc))
    body = upload(client, png_bytes(), model="stub_b").json()
    assert svc.calls[-1] == "stub_b" and body["model"] == "stub_b"
    assert body["top"][0]["disease"] != "Cardiomegaly"
    assert upload(client, png_bytes()).json()["model"] == "stub_a"  # default


def test_unknown_model_is_rejected(client):
    assert upload(client, png_bytes(), model="nope").status_code == 400
    r = upload(client, png_bytes(), "/predict/visualize", disease="Mass", model="nope")
    assert r.status_code == 400


def test_compare_returns_every_model_with_heatmaps(client):
    body = upload(client, png_bytes(), "/predict/compare", disease="Mass").json()
    assert [r["model"] for r in body["results"]] == ["stub_a", "stub_b", "ensemble"]
    assert all("overlay_png_base64" in r for r in body["results"])
    plain = upload(client, png_bytes(), "/predict/compare").json()
    assert all("overlay_png_base64" not in r for r in plain["results"])


def test_hub_serves_registry_models_with_temperature_and_ensemble(tmp_path):
    """Real ModelHub on two tiny ONNX models listed in a registry."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("onnxruntime")
    import json

    from api.service import ModelHub

    class Tiny(torch.nn.Module):
        def __init__(self, bias):
            super().__init__()
            self.bias = bias

        def forward(self, x):
            return x.mean(dim=(2, 3))[:, :1].repeat(1, 14) * 0 + self.bias

    entries = []
    for i, (bias, temp) in enumerate([(2.0, 1.0), (-2.0, 2.0)]):
        path = tmp_path / f"m{i}.onnx"
        torch.onnx.export(Tiny(bias), torch.randn(1, 3, 224, 224), str(path),
                          input_names=["image"], output_names=["logits"], dynamo=False)
        entries.append({"id": f"m{i}", "arch": "densenet121", "onnx": path.name,
                        "temperature": temp, "default": i == 1})
    entries.append({"id": "missing", "arch": "resnet50", "onnx": "nope.onnx"})
    (tmp_path / "registry.json").write_text(json.dumps({"models": entries}), encoding="utf-8")

    hub = ModelHub(tmp_path / "registry.json")
    assert hub.model_ids == ["m0", "m1", "ensemble"] and hub.default_id == "m1"
    img = Image.fromarray(np.zeros((64, 64), dtype=np.uint8))
    p0, p1 = hub.predict(img, "m0"), hub.predict(img, "m1")
    assert np.allclose(p0, 1 / (1 + np.exp(-2.0)))
    assert np.allclose(p1, 1 / (1 + np.exp(1.0)))  # logit -2 divided by T = 2
    assert np.allclose(hub.predict(img, "ensemble"), (p0 + p1) / 2)
