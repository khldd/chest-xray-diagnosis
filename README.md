# Chest X-ray Diagnosis

Multi-label diagnosis of 14 thoracic diseases from chest X-rays (NIH Chest X-ray 14), using DenseNet121 transfer learning with Grad-CAM explanations.

End-of-module project for **Deep Learning Avancé** (TEK-UP, ING-5, 2026-2027), also submitted for **Big Data**.

**Team:** Hayder Methneni, Khaled Ayedi

> Research and educational demo. Not a diagnostic tool and not medical advice.

## Current results

| Model | Split | Mean AUC (14 classes) |
|---|---|---|
| DenseNet121, ImageNet init, full fine-tuning, weighted BCE | Validation (12,608 images) | 0.831 |
| Same | Test (25,596 images) | 0.7987 |

Per-class test AUC is in [`results/test_metrics.json`](results/test_metrics.json) (best: Hernia 0.933, Emphysema 0.909, Cardiomegaly 0.871; weakest: Pneumonia 0.688, Infiltration 0.693). The gap to validation is expected: the best epoch was selected on validation, and the test patients are different.

Prototype: [`notebooks/chest-xray-diagnosis.ipynb`](notebooks/chest-xray-diagnosis.ipynb), trained on a Kaggle T4 ([Kaggle notebook](https://www.kaggle.com/code/haydermethneni/chest-xray-diagnosis)).

## Dataset

[NIH Chest X-ray 14](https://www.kaggle.com/datasets/nih-chest-xrays/data): 112,120 frontal X-rays from 30,805 patients, about 45 GB. Not stored in this repo.

- Official split: `train_val_list.txt` (86,524) / `test_list.txt` (25,596).
- Validation: 15% of train_val, split by `Patient ID` with `GroupShuffleSplit(random_state=42)` so no patient is in both train and validation.
- Labels: 14 diseases, multi-hot. "No Finding" is the all-zero vector.

## Model weights

Weights are versioned with [DVC](https://dvc.org/): git only holds `models/best_model.pth.dvc`. No DVC remote is configured yet, so download the weights from Kaggle (needs access to the model and the [Kaggle CLI](https://github.com/Kaggle/kaggle-api)):

```bash
python -m kaggle models instances versions download haydermethneni/best-model/pytorch/default/1 -p models --untar
```

`best_model.pth` is a plain `state_dict` for `torchvision.models.densenet121` with `classifier = nn.Linear(1024, 14)`. Its md5 should match the one in `models/best_model.pth.dvc`.

The API serves an ONNX export of it. Generate that from the `.pth`:

```bash
python -m src.export_onnx --weights models/best_model.pth   # writes best_model.onnx + best_model.onnx.data
```

The exporter writes the weights to a separate `best_model.onnx.data` file. Keep it next to the `.onnx`. On Windows, set `PYTHONIOENCODING=utf-8` first, otherwise torch's log output can crash the console.

## Pipeline usage

```bash
pip install -r requirements/dev.txt
python -m src.train --config configs/base.yaml            # train (MLflow-tracked, resumable)
python -m src.evaluate --weights models/best_model.pth --split test --out results/test_metrics.json
python -m src.export_onnx --weights models/best_model.pth  # ONNX export + parity check
pytest && ruff check .
```

Experiments override `configs/base.yaml` (loss, augmentation, `freeze_backbone`, epochs); see `CONTRIBUTING.md` for the branch and commit workflow. `notebooks/kaggle_runner.ipynb` runs the evaluation and ablations on Kaggle.

## Running the app

The app needs `models/best_model.onnx`, `models/best_model.onnx.data` and `models/best_model.pth` (the `.pth` is used for Grad-CAM). See "Model weights".

### Docker Compose

```bash
docker compose up --build
```

Open http://localhost:8080. The API is also on http://localhost:8000 (interactive docs at `/docs`). The `models/` folder is mounted read-only into the API container; weights are never baked into an image.

### Local development

```bash
# terminal 1: API
pip install -r requirements/api.txt
uvicorn api.main:app --port 8000

# terminal 2: frontend (proxies /api to localhost:8000)
cd frontend
npm install
npm run dev
```

Open http://localhost:5173. The first Grad-CAM heatmap after the API starts takes 7-13 seconds (torch and model loading); later ones take about 0.3 s. `/predict` takes about 25 ms per image on CPU.

### API

| Endpoint | Purpose |
|---|---|
| `GET /health` | Status, whether the model is loaded |
| `GET /model/metadata` | Classes, input size, validation/test AUC, ONNX hash |
| `POST /predict` | Upload an image (`file`); returns the 14 probabilities, top 5 and latency |
| `POST /predict/visualize?disease=X` | Same, plus a Grad-CAM overlay as base64 PNG |

Uploads are decoded for real: non-images get a 400 and files over 10 MB a 413.

## Roadmap

Following the course milestones:

- [x] Topic and dataset chosen, repo initialized, DVC set up
- [ ] DVC remote
- [x] `src/` pipeline (dataset, architecture, training with MLflow, evaluation + ONNX export, Grad-CAM)
- [x] Test-set evaluation
- [ ] Baseline comparison, augmentation and loss ablations (configs ready, runs pending)
- [x] FastAPI backend (`/health`, `/model/metadata`, `/predict`, `/predict/visualize`) with ONNX Runtime
- [x] React frontend (upload, example gallery, webcam, probability bars, Grad-CAM overlay, latency)
- [x] Docker Compose, GitHub Actions (lint, pytest, docker build)
- [ ] LaTeX technical report
