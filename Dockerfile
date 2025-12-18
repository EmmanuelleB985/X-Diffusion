# Multi-stage build for X-Diffusion
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04 AS builder

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0;7.5;8.0;8.6;8.9;9.0"
ENV CUDA_HOME=/usr/local/cuda

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.9 \
    python3.9-dev \
    python3-pip \
    git \
    wget \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create working directory
WORKDIR /app

# Copy requirements files
COPY requirements.txt requirements-dev.txt ./

# Install Python dependencies
RUN python3.9 -m pip install --upgrade pip setuptools wheel && \
    python3.9 -m pip install --no-cache-dir -r requirements.txt

# Production stage
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PATH="/app/.local/bin:${PATH}"

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    python3.9 \
    python3-pip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 xdiffusion && \
    mkdir -p /app && \
    chown -R xdiffusion:xdiffusion /app

# Switch to non-root user
USER xdiffusion
WORKDIR /app

# Copy Python packages from builder
COPY --from=builder --chown=xdiffusion:xdiffusion /usr/local/lib/python3.9 /usr/local/lib/python3.9
COPY --from=builder --chown=xdiffusion:xdiffusion /usr/local/bin /usr/local/bin

# Copy application code
COPY --chown=xdiffusion:xdiffusion scripts/ ./scripts/
COPY --chown=xdiffusion:xdiffusion DXAtoMRI/ ./DXAtoMRI/
COPY --chown=xdiffusion:xdiffusion configs/ ./configs/
COPY --chown=xdiffusion:xdiffusion requirements.txt ./

# Create necessary directories
RUN mkdir -p checkpoints data results logs

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python3.9 -c "import torch; print('Health check passed')" || exit 1

# Default command
ENTRYPOINT ["python3.9"]
CMD ["scripts/inference.py", "--help"]

# Labels
LABEL maintainer="X-Diffusion Team"
LABEL version="1.0.0"
LABEL description="X-Diffusion: Generating 3D MRI Volumes from Single Images"
