# SAM3 RunPod Serverless Worker
# Downloads model from Hugging Face at runtime (smaller image)
#
# Build: docker build -t username/sam3-runpod:latest .
# Push:  docker push username/sam3-runpod:latest

FROM runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04

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

# Clone SAM3 repository from Meta and install
RUN git clone https://github.com/facebookresearch/sam3.git /app/sam3-repo && \
    cd /app/sam3-repo && \
    pip install --no-cache-dir -e .

# Copy the worker script
COPY runpod_worker.py /app/runpod_worker.py

# Verify SAM3 is installed
RUN python -c "import sam3; print('✅ SAM3 installed successfully')"

# Install PyTorch with CUDA 12.8 (at end to maximize cache usage)
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://mirrors.nju.edu.cn/pytorch/whl/cu128

# Start the serverless worker
CMD ["python", "-u", "/app/runpod_worker.py"]
