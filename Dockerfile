# Base image with Python 3.12 slim
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    ffmpeg \
    libsm6 \
    libxext6 \
    wget \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA 12.8 support
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Install transformers from GitHub
RUN pip install --no-cache-dir git+https://github.com/huggingface/transformers

# Clone and install sam3 from Facebook Research
RUN git clone https://github.com/facebookresearch/sam3.git /app/sam3 && \
    cd /app/sam3 && \
    pip install -e .

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py .

# Set the entrypoint for RunPod serverless
CMD ["python", "-u", "app.py"]
