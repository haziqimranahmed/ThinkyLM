# ThinkyLM — CPU-Compatible Dockerfile
# ======================================
# Builds the FastAPI inference server.
# No datasets or checkpoints are included in the image.
# Mount a checkpoint at runtime:
#   docker run -v /path/to/checkpoints:/app/checkpoints \
#              -e THINKYLM_CHECKPOINT=/app/checkpoints/debug_1m/debug_1m_step000020 \
#              -p 8000:8000 thinkylm:latest
#
# Build: docker build -t thinkylm:latest .
# Run:   see above

FROM python:3.11-slim

LABEL maintainer="Haziq Imran"
LABEL description="ThinkyLM FastAPI inference server"
LABEL version="0.1.0"

# Set working directory
WORKDIR /app

# Install CPU-only PyTorch and dependencies
# This keeps the image lean — no CUDA drivers
COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir \
        tokenizers \
        pyyaml \
        pydantic \
        fastapi \
        "uvicorn[standard]" \
        tqdm \
        safetensors \
        psutil

# Copy source code (no data or checkpoints — mount at runtime)
COPY thinkylm/ ./thinkylm/
COPY api/ ./api/
COPY inference/ ./inference/
COPY data_pipeline/ ./data_pipeline/
COPY configs/ ./configs/
COPY tokenizer/sample_text.txt ./tokenizer/sample_text.txt

# Environment defaults
ENV TOKENIZERS_PARALLELISM=false
ENV OMP_NUM_THREADS=2
ENV MKL_NUM_THREADS=2
ENV API_HOST=0.0.0.0
ENV API_PORT=8000
ENV THINKYLM_CONFIG=configs/debug_1m.yaml

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# Run the API server
CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
