import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from src.evaluate import auc_per_class  # noqa: E402
from src.models.densenet import build_model  # noqa: E402


def test_model_output_shape():
    model = build_model(pretrained=False).eval()
    with torch.no_grad():
        assert model(torch.randn(2, 3, 224, 224)).shape == (2, 14)


def test_freeze_backbone_trains_only_head():
    model = build_model(pretrained=False, freeze_backbone=True)
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    assert trainable == {"classifier.weight", "classifier.bias"}


def test_state_dict_keys_match_prototype_checkpoint():
    keys = build_model(pretrained=False).state_dict().keys()
    assert "classifier.weight" in keys and "features.norm5.weight" in keys


def test_auc_perfect_and_nan_for_missing_class():
    labels = np.zeros((6, 14), dtype=np.float32)
    labels[:3, 0] = 1  # only class 0 has both positives and negatives
    probs = np.random.default_rng(0).random((6, 14)).astype(np.float32)
    probs[:3, 0] += 1.0  # positives ranked above negatives -> AUC 1.0
    aucs, mean_auc = auc_per_class(probs, labels)
    assert aucs["Atelectasis"] == 1.0
    assert np.isnan(aucs["Hernia"])
    assert mean_auc == 1.0  # nanmean ignores classes without positives


def test_onnx_export_parity(tmp_path):
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    from src.export_onnx import export

    weights = tmp_path / "w.pth"
    torch.save(build_model(pretrained=False).state_dict(), weights)
    assert export(str(weights), str(tmp_path / "m.onnx")) < 1e-3
