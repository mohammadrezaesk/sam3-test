# SAM3 RunPod Serverless Worker - Self-Contained Build
# Downloads SAM3 from GitHub and model weights from Hugging Face
#
# Build: docker build --build-arg HF_TOKEN=your_token -t ghcr.io/username/sam3-runpod:latest .
# Push:  docker push ghcr.io/username/sam3-runpod:latest

FROM runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04

# Build arguments
ARG HF_TOKEN
ENV HF_TOKEN=${HF_TOKEN}

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV SAM3_MODEL_DIR=/app/sam3-files
ENV CONFIDENCE_THRESHOLD=0.5

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    git-lfs \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/* \
    && git lfs install

# Copy and install Python dependencies from requirements.txt
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --ignore-installed -r requirements.txt

# Install transformers from GitHub (required for SAM3)
RUN pip install --no-cache-dir git+https://github.com/huggingface/transformers

# Clone SAM3 repository from Meta
RUN git clone https://github.com/facebookresearch/sam3.git /app/sam3-repo && \
    cd /app/sam3-repo && \
    pip install --no-cache-dir -e .

# Download model files from Hugging Face
RUN python << 'EOF'
from huggingface_hub import snapshot_download, login
import os
token = os.environ.get('HF_TOKEN')
if token:
    login(token=token)
snapshot_download(repo_id='facebook/sam3', local_dir='/app/sam3-files', local_dir_use_symlinks=False)
print('✅ Model downloaded successfully')
EOF

# Copy the worker script (create this file in the same directory as Dockerfile)
COPY runpod_worker.py /app/runpod_worker.py

# Verify everything is in place
RUN ls -la /app/sam3-files/ && \
    python -c "import sam3; print('✅ SAM3 imported successfully')" && \
    echo "✅ Build complete!"

# Install PyTorch with CUDA 12.8 (at end to maximize cache usage)
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://mirrors.nju.edu.cn/pytorch/whl/cu128

# Start the serverless worker
CMD ["python", "-u", "/app/runpod_worker.py"]
