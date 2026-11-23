# Build from the repo root:  docker build -f api/Dockerfile -t chest-xray-api .
# Model files are NOT baked in: mount the models/ folder at /models (it needs best_model.onnx,
# best_model.onnx.data and best_model.pth, all three).
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# grad-cam depends on opencv, which needs these system libraries even on a headless server.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home app

WORKDIR /app

# CPU-only torch is far smaller than the default CUDA wheel; install it first so the
# requirements file below treats torch/torchvision as already satisfied.
COPY requirements/api.txt requirements/api.txt
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install -r requirements/api.txt

COPY src ./src
COPY api ./api
COPY results ./results

ENV MODEL_ONNX=/models/best_model.onnx \
    MODEL_PTH=/models/best_model.pth \
    TEST_METRICS=/app/results/test_metrics.json

USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
