# CUDA runtime base (good for Cloud Run GPU / L4)
FROM nvidia/cuda:12.2.2-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/tmp/hf \
    TRANSFORMERS_CACHE=/tmp/hf \
    TOKENIZERS_PARALLELISM=false \
    PORT=8080

# System deps (openblas helps faiss-cpu; git sometimes needed for model code)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv \
    git curl ca-certificates \
    libopenblas0 libgomp1 \
 && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip

# Install CUDA-enabled PyTorch first (so downstream deps don't pull CPU torch)
# cu121 wheels are widely available and work fine on CUDA 12.x runtimes
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu121

WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy app + your artifacts
COPY app.py /app/app.py
COPY artifacts/ /app/artifacts/
COPY data/ /app/data/

EXPOSE 8080
CMD ["python3", "app.py"]
