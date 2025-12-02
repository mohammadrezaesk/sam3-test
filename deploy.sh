#!/bin/bash
set -e

if [ -z "$HF_TOKEN" ]; then
    echo "Error: HF_TOKEN environment variable is not set"
    exit 1
fi

python3 -m venv venv --system-site-packages
source venv/bin/activate

if [ ! -d "sam3" ]; then
    git clone https://github.com/facebookresearch/sam3.git
fi

cd sam3
pip install -e .
cd ..

pip install git+https://github.com/huggingface/transformers.git
pip install gradio pillow numpy matplotlib einops decord pycocotools opencv-python scikit-image scikit-learn pandas tqdm accelerate av

export HF_TOKEN=$HF_TOKEN
export PORT=${PORT:-7860}

python app.py

