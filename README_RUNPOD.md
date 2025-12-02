# SAM3 Remote GPU Setup with RunPod Serverless

This guide explains how to deploy SAM3 as a serverless endpoint on RunPod and use it from your local machine.

Uses the official [RunPod Python SDK](https://github.com/runpod/runpod-python).

## Architecture

```
┌─────────────────┐         ┌──────────────────────────┐
│  Your Machine   │         │  RunPod Serverless       │
│  (Streamlit UI) │ ◄─────► │  (SAM3 Worker)           │
│                 │  HTTPS  │  - Auto-scaling          │
│  app.py         │         │  - Pay per second        │
│  runpod_client  │         │  - GPU (A100, etc.)      │
└─────────────────┘         └──────────────────────────┘
```

## Quick Start

### 1. Create RunPod Account & Get API Key

1. Go to [RunPod](https://www.runpod.io/) and create an account
2. Navigate to **Settings** → **API Keys**
3. Create a new API key and save it

### 2. Build Docker Image for RunPod

Create a `Dockerfile` for your worker:

```dockerfile
FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install runpod

# Copy SAM3 code and model files
COPY sam3/ ./sam3/
COPY sam3-files/ ./sam3-files/
COPY runpod_worker.py .

# Set environment variables
ENV SAM3_MODEL_DIR=/app/sam3-files
ENV CONFIDENCE_THRESHOLD=0.5

# Start the worker
CMD ["python", "-u", "runpod_worker.py"]
```

Build and push to Docker Hub:

```bash
docker build -t yourusername/sam3-runpod:latest .
docker push yourusername/sam3-runpod:latest
```

### 3. Create Serverless Endpoint on RunPod

1. Go to **RunPod Console** → **Serverless**
2. Click **+ New Endpoint**
3. Configure:
   - **Name**: `sam3-segmentation`
   - **Docker Image**: `yourusername/sam3-runpod:latest`
   - **GPU**: Select appropriate GPU (A100 recommended)
   - **Min Workers**: 0 (scale to zero when idle)
   - **Max Workers**: 5 (adjust based on needs)
4. Click **Create**
5. Copy the **Endpoint ID** (e.g., `abc123xyz`)

### 4. Configure Your Local Machine

```bash
# Copy the example environment file
cp .env.example .env

# Edit .env with your credentials
nano .env
```

Set these values:
```bash
USE_REMOTE_GPU=true
RUNPOD_ENDPOINT_ID=your_endpoint_id_here
RUNPOD_API_KEY=your_api_key_here
```

### 5. Run the Application

```bash
# Install dependencies
pip install -r requirements.txt

# Run Streamlit app
streamlit run app.py
```

## Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `USE_REMOTE_GPU` | Enable remote GPU mode | Yes |
| `RUNPOD_ENDPOINT_ID` | Your serverless endpoint ID | Yes |
| `RUNPOD_API_KEY` | Your RunPod API key | Yes |
| `SAM3_MODEL_DIR` | Path to model files (local mode) | No |
| `CONFIDENCE_THRESHOLD` | Detection threshold (0-1) | No |

## Using the Python Client Directly

```python
import runpod
from runpod_client import RunPodClient
from PIL import Image

# Initialize client
client = RunPodClient(
    endpoint_id="your_endpoint_id",
    api_key="your_api_key"
)

# Segment an image
image = Image.open("test.jpg")
result = client.segment_image(image, prompt="person")

print(f"Found {result.num_objects} objects")
for i, score in enumerate(result.scores):
    print(f"  Object {i}: confidence {score:.3f}")

# Access masks and boxes
for mask in result.masks:
    print(f"  Mask shape: {mask.shape}")
```

## Worker Input/Output Format

### Image Segmentation

**Input:**
```json
{
    "input": {
        "task": "image",
        "image_base64": "<base64-encoded-png>",
        "prompt": "person",
        "confidence_threshold": 0.5
    }
}
```

**Output:**
```json
{
    "masks": [[[0, 0, 1, 1, ...], ...], ...],
    "boxes": [[x1, y1, x2, y2], ...],
    "scores": [0.95, 0.87, ...],
    "num_objects": 2
}
```

### Video Segmentation

**Input:**
```json
{
    "input": {
        "task": "video",
        "frames_base64": ["<frame1>", "<frame2>", ...],
        "prompt": "person",
        "max_frames": 30
    }
}
```

**Output:**
```json
{
    "num_frames_processed": 30,
    "outputs_per_frame": {
        "0": {"masks": [...], "boxes": [...], "object_ids": [...]},
        "1": {"masks": [...], "boxes": [...], "object_ids": [...]},
        ...
    }
}
```

## Local Testing

Test the worker locally before deploying:

```bash
# Run worker in test mode
python runpod_worker.py --rp_serve_api

# In another terminal, send test request
curl -X POST http://localhost:8000/runsync \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
        "task": "image",
        "image_base64": "'$(base64 -w0 test.jpg)'",
        "prompt": "person"
    }
}'
```

## Cost Optimization

### Pay-Per-Second Billing
RunPod charges only when your endpoint is processing requests. With `Min Workers: 0`, you pay nothing when idle.

### Cold Start
First request after idle period takes ~30-60 seconds (model loading). Subsequent requests are fast.

### Tips
- Use `Min Workers: 1` for always-on, faster response
- Batch multiple images in one request if possible
- Choose appropriate GPU (A40 is cost-effective, A100 is fastest)

## Troubleshooting

### "Endpoint not found"
- Verify your `RUNPOD_ENDPOINT_ID` is correct
- Check that the endpoint is active in RunPod Console

### "Authentication failed"
- Verify your `RUNPOD_API_KEY` is correct
- Regenerate API key if needed

### "Model loading failed"
- Ensure `sam3-files/` is included in your Docker image
- Check that `sam3.pt` and `config.json` exist

### Timeout errors
- Video processing can take several minutes
- Increase timeout: `client = RunPodClient(..., timeout=600)`

### GPU out of memory
- Use a GPU with more VRAM
- Reduce `max_frames` for video processing

## Files Reference

| File | Description |
|------|-------------|
| `runpod_worker.py` | Serverless worker (deploy to RunPod) |
| `runpod_client.py` | Client using RunPod SDK |
| `app.py` | Streamlit UI with remote GPU support |
| `.env.example` | Environment variable template |
