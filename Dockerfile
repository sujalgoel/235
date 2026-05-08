# Backend container for the RealityCheck FastAPI service.
# The frontend ships separately as a static bundle (Vite -> dist/), so this
# image only carries the Python runtime and the model dependencies.

FROM python:3.13-slim AS base

# System libs needed by opencv, exiftool, and torchvision image codecs.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
        libjpeg-dev zlib1g-dev exiftool curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first so the layer cache survives source edits.
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY config ./config
COPY src ./src

# Pre-create the writable directories the app expects.
RUN mkdir -p logs uploads visualizations models

EXPOSE 8000

# Use a single worker by default; production overrides via env.
ENV ENVIRONMENT=development \
    API_HOST=0.0.0.0 \
    API_PORT=8000 \
    PYTHONUNBUFFERED=1

HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD curl -fsS http://localhost:${API_PORT}/health || exit 1

CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
