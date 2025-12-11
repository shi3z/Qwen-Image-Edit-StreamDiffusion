# Dockerfile for Qwen-Image-Edit-StreamDiffusion
# Optimized for NVIDIA GB10 (Blackwell, SM 12.1) with CUDA 12.9
#
# Build: docker build -t qwen-edit .
# Run:   docker run --gpus all -p 8086:8086 qwen-edit

FROM nvcr.io/nvidia/pytorch:25.04-py3

# Set working directory
WORKDIR /app

# Environment variables for GB10 compatibility
ENV PYTORCH_JIT=0
ENV TORCH_COMPILE_DISABLE=1
ENV CUDA_MODULE_LOADING=LAZY
ENV TORCH_CUDA_ARCH_LIST="12.0;12.1"

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY pyproject.toml ./

# Install Python dependencies
# Note: NVIDIA container already has PyTorch, we just need diffusers and other deps
RUN pip install --no-cache-dir \
    diffusers>=0.32.0 \
    transformers>=4.47.0 \
    accelerate>=1.2.1 \
    safetensors \
    pillow \
    numpy \
    fastapi \
    uvicorn \
    peft

# Copy application code
COPY . .

# Expose the API port
EXPOSE 8086

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8086/health || exit 1

# Run the server
CMD ["python", "server.py"]
