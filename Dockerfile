# Base image with CUDA support for RunPod
FROM runpod/pytorch:2.4.0-py3.12-cuda12.4.1-devel-ubuntu22.04

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
    && rm -rf /var/lib/apt/lists/*

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

