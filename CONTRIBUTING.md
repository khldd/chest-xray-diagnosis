# Contributing

Two-person team: Khaled Ayedi (code and infrastructure) and Hayder Methneni (experiments and results).

## Branches
- `main` only receives merged pull requests.
- One branch per piece of work: `feat/*`, `exp/*`, `docs/*`, `chore/*`.
- Merge with a merge commit (not squash) so individual commits stay visible.
- Every PR gets a review from the other teammate.

## Commit messages
Conventional commits: `feat:`, `fix:`, `test:`, `docs:`, `ci:`, `chore:`, `refactor:`, `exp:`, `data:`.
Keep the subject under about 70 characters and say *why* in the body when it is not obvious.

## Local setup
```bash
python -m venv .venv
pip install -r requirements/dev.txt
ruff check .
pytest
```

## Data and weights
- The NIH dataset is never committed. Weights and processed data go through DVC.
- Never commit `kaggle.json`, `.env` or `*.pth` / `*.onnx` files.
