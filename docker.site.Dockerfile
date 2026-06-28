FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Minimal deps for serve_fastapi.py and data loading
RUN pip install --upgrade pip \
    && pip install \
        fastapi==0.115.6 \
        uvicorn[standard]==0.30.6 \
        pydantic==2.12.1 \
        numpy==1.26.4 \
        pandas==2.2.3 \
        pyarrow==18.1.0 \
        tensorflow==2.16.1 \
        tf_keras==2.16.0 \
        requests==2.32.3

# CPU-only PyTorch for transformers sentiment pipeline inside API container.
RUN pip install --index-url https://download.pytorch.org/whl/cpu \
        torch==2.6.0

RUN pip install \
        transformers==4.49.0 \
        sentencepiece==0.2.0 \
        safetensors==0.5.3

WORKDIR /app
