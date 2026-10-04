# Chest X-ray Diagnosis

Multi-label diagnosis of 14 thoracic diseases from chest X-rays (NIH Chest X-ray 14), using DenseNet121 transfer learning with Grad-CAM explanations.

End-of-module project for **Deep Learning Avancé** (TEK-UP, ING-5, 2026-2027), also submitted for **Big Data**.

**Team:** Hayder Methneni, Khaled Ayedi

## Current results

| Model | Split | Mean AUC (14 classes) |
|---|---|---|
| DenseNet121, ImageNet init, full fine-tuning, weighted BCE | Validation (12,608 images) | 0.831 |
| Same | Test (25,596 images) | not evaluated yet |

Prototype: [`notebooks/chest-xray-diagnosis.ipynb`](notebooks/chest-xray-diagnosis.ipynb), trained on a Kaggle T4 ([Kaggle notebook](https://www.kaggle.com/code/haydermethneni/chest-xray-diagnosis)).

## Dataset

[NIH Chest X-ray 14](https://www.kaggle.com/datasets/nih-chest-xrays/data): 112,120 frontal X-rays from 30,805 patients, about 45 GB. Not stored in this repo.

- Official split: `train_val_list.txt` (86,524) / `test_list.txt` (25,596).
- Validation: 15% of train_val, split by `Patient ID` with `GroupShuffleSplit(random_state=42)` so no patient is in both train and validation.
- Labels: 14 diseases, multi-hot. "No Finding" is the all-zero vector.

## Model weights

Weights are versioned with [DVC](https://dvc.org/): git only holds `models/best_model.pth.dvc`. No DVC remote is configured yet, so download the weights from Kaggle (needs access to the private model and the [Kaggle CLI](https://github.com/Kaggle/kaggle-api)):

```bash
python -m kaggle models instances versions download haydermethneni/best-model/pytorch/default/1 -p models --untar
```

`best_model.pth` is a plain `state_dict` for `torchvision.models.densenet121` with `classifier = nn.Linear(1024, 14)`.

## Roadmap

Following the course milestones:

- [x] Topic and dataset chosen, repo initialized, DVC set up
- [ ] DVC remote
- [ ] `src/` pipeline (dataset, architecture, training with MLflow, evaluation + ONNX export, Grad-CAM)
- [ ] Test-set evaluation, baseline comparison, augmentation and loss ablations
- [ ] FastAPI backend (`/health`, `/model/metadata`, `/predict`, `/predict/visualize`) with ONNX Runtime
- [ ] React frontend (upload, example gallery, webcam, probability bars, Grad-CAM overlay, latency)
- [ ] Docker Compose, GitHub Actions (lint, pytest, docker build)
- [ ] LaTeX technical report
