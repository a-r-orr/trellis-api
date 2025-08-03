# 1. Use the stable and widely supported Ubuntu 22.04 base image
FROM nvidia/cuda:12.8.0-devel-ubuntu22.04

# 2. Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# 3. Install Python 3.10 and essential build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3-pip \
    python3.10-venv \
    python-is-python3 \
    git \
    build-essential \
    libx11-6 \
    libgl1 \
    libglib2.0-0 \
    libjpeg-dev \
    zlib1g-dev \
    libpng-dev \
    && rm -rf /var/lib/apt/lists/*

# 4. Install PyTorch for CUDA 12.8
# Using python3.10 -m pip to ensure the correct pip is used
RUN python3.10 -m pip install --no-cache-dir --upgrade pip && \
    python3.10 -m pip install --no-cache-dir \
    "torch==2.7.1+cu128" \
    "torchvision==0.22.1+cu128" \
    "torchaudio==2.7.1+cu128" \
    --index-url https://download.pytorch.org/whl/cu128

# 5. Install Pillow-SIMD system-wide
RUN pip uninstall -y pillow || true && pip install --no-cache-dir pillow-simd

# 6. Create a non-root user and set up the app environment
WORKDIR /usr/src/app
RUN useradd --create-home appuser

ENV CUDA_HOME=/usr/local/cuda
ENV TORCH_CUDA_ARCH_LIST="12.0"
ENV NVIDIA_DISABLE_REQUIRE=1

USER appuser
ENV PATH="/home/appuser/.local/bin:${PATH}"

# 7. Copy and run the setup script
COPY --chown=appuser:appuser extensions/vox2seq/ /tmp/extensions/vox2seq/
COPY --chown=appuser:appuser setup.docker.sh .
RUN chmod +x setup.docker.sh && ./setup.docker.sh

# 8. Copy application code and startup script
COPY --chown=appuser:appuser src/ ./src
COPY --chown=appuser:appuser trellis/ ./trellis
COPY --chown=appuser:appuser configs/ ./configs
COPY --chown=appuser:appuser helper_script.sh ./
RUN chmod +x helper_script.sh

# 9. Expose port and run
EXPOSE 8080
CMD ["./helper_script.sh"]