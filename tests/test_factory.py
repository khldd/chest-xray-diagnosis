"""Every architecture in the factory: head, freezing, Grad-CAM layer, ONNX parity."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from src.models.factory import ARCHS, build_model, count_parameters, get_arch  # noqa: E402

NAMES = sorted(ARCHS)


@pytest.mark.parametrize("name", NAMES)
def test_output_shape(name):
    model = build_model(name, pretrained=False).eval()
    with torch.no_grad():
        assert model(torch.randn(2, 3, 224, 224)).shape == (2, 14)


@pytest.mark.parametrize("name", NAMES)
def test_freeze_leaves_only_the_head(name):
    model = build_model(name, pretrained=False, freeze_backbone=True)
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    assert trainable and all(n.startswith(get_arch(name).head_prefix) for n in trainable)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert n_trainable < count_parameters(model) / 100


@pytest.mark.parametrize("name", NAMES)
def test_gradcam_layer_gives_a_heatmap(name):
    pytest.importorskip("pytorch_grad_cam")
    from src.gradcam import gradcam_heatmaps

    model = build_model(name, pretrained=False).eval()
    heat = gradcam_heatmaps(model, torch.randn(2, 3, 224, 224), [0, 5], name)
    assert heat.shape == (2, 224, 224)
    assert np.isfinite(heat).all() and heat.min() >= 0


def test_unknown_architecture():
    with pytest.raises(ValueError):
        build_model("vgg_lol", pretrained=False)


@pytest.mark.parametrize("name", NAMES)
def test_onnx_parity(name, tmp_path):
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    from src.export_onnx import export

    weights = tmp_path / "w.pth"
    torch.save(build_model(name, pretrained=False).state_dict(), weights)
    assert export(str(weights), str(tmp_path / "m.onnx"), arch=name) < 1e-3
