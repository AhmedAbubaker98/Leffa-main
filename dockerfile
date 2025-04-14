# Use BuildKit explicitly for parallel layer processing
# syntax=docker/dockerfile:1.4
FROM nvidia/cuda:12.1.1-base-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PIP_NO_CACHE_DIR=1 \
    DOCKER_BUILDKIT=1 \
    TORCH_CUDA_ARCH_LIST="8.0"  
    # Disable GPU arch compilation

# Use faster APT mirrors and install minimal runtime deps
RUN sed -i 's/archive.ubuntu.com/mirror.rackspace.com/g' /etc/apt/sources.list && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.10 \
        python3-pip \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install PyTorch with pre-built CUDA binaries first
# Using torch-nightly for latest optimized binaries
COPY requirements.txt .
RUN pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121 && \
    pip install -r requirements.txt && \
    find /usr/local/lib -type d -name '__pycache__' -exec rm -rf {} + && \
    find /usr/local/lib -type d -name 'tests' -exec rm -rf {} +

# Copy application code (exclude ckpts via .dockerignore)
COPY . .

# Runtime config
VOLUME /app/ckpts  # Mount models at runtime
EXPOSE 8000
CMD ["uvicorn", "leffa_api:app", "--host", "0.0.0.0", "--port", "8000"]