# Use a slim Python base image
FROM python:3.9-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create and activate virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install torch with CUDA support (adjust based on your CUDA version)
RUN pip install --no-cache-dir torch torchvision --extra-index-url https://download.pytorch.org/whl/cu118

# Final stage - create a minimal image
FROM python:3.9-slim

# Copy only necessary files from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Create working directory
WORKDIR /app

# Copy application code
COPY leffa leffa/
COPY preprocess preprocess/
COPY leffa_utils leffa_utils/
COPY app.py leffa_api.py ./

# Create directories that will be mounted at runtime
RUN mkdir -p /app/ckpts /app/temp

# Expose API port
EXPOSE 8000

# Set the entrypoint
CMD ["uvicorn", "leffa_api:app", "--host", "0.0.0.0", "--port", "8000"]