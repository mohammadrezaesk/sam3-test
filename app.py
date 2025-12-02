import runpod
import requests
import traceback
import sys
import os
from io import BytesIO
from PIL import Image

# Initialize model and processor as None
model = None
processor = None

def initialize_model():
    """Initialize the SAM3 model and processor."""
    global model, processor
    try:
        print("Loading SAM3 model...", flush=True)
        
        # Read HuggingFace token from environment
        hf_token = os.getenv("HF_TOKEN")
        if hf_token:
            print("HF_TOKEN found in environment", flush=True)
            # Set the token for HuggingFace Hub (multiple ways for compatibility)
            os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token
            os.environ["HF_TOKEN"] = hf_token
            
            # Try to login to HuggingFace Hub if the library is available
            try:
                from huggingface_hub import login
                login(token=hf_token, add_to_git_credential=False)
                print("Logged in to HuggingFace Hub", flush=True)
            except ImportError:
                print("huggingface_hub not available, using environment variable", flush=True)
            except Exception as e:
                print(f"Warning: Could not login to HuggingFace Hub: {str(e)}", flush=True)
        else:
            print("Warning: HF_TOKEN not found in environment", flush=True)
        
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
        
        # Pass token to model builder if it accepts it
        # If build_sam3_image_model doesn't accept token directly, 
        # the token will be used via environment variables
        if hf_token:
            try:
                model = build_sam3_image_model(token=hf_token)
            except TypeError:
                # If token parameter is not supported, try without it
                # The token should be picked up from environment
                model = build_sam3_image_model()
        else:
            model = build_sam3_image_model()
        
        processor = Sam3Processor(model)
        print("Model loaded successfully!", flush=True)
    except Exception as e:
        print(f"Error loading model: {str(e)}", flush=True)
        print(traceback.format_exc(), flush=True)
        sys.exit(1)

def download_image(url):
    """Download image from URL."""
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content))
        return image
    except Exception as e:
        print(f"Error downloading image: {str(e)}", flush=True)
        raise

def handler(job):
    """Handle the job request."""
    try:
        if model is None or processor is None:
            raise RuntimeError("Model not initialized")
        
        job_input = job.get("input", {})
        image_url = job_input.get("image_url")
        
        if not image_url:
            return {"error": "Missing 'image_url' in job input"}
        
        print(f"Processing image from: {image_url}", flush=True)
        image = download_image(image_url)
        prompt = job_input.get("prompt", "person")
        
        print(f"Setting image and processing prompt: {prompt}", flush=True)
        inference_state = processor.set_image(image)
        output = processor.set_text_prompt(state=inference_state, prompt=prompt)
        
        print("Processing completed successfully", flush=True)
        return output
    except Exception as e:
        error_msg = f"Error in handler: {str(e)}"
        print(error_msg, flush=True)
        print(traceback.format_exc(), flush=True)
        return {"error": error_msg}

# Initialize model before starting serverless
try:
    initialize_model()
except Exception as e:
    print(f"Failed to initialize model: {str(e)}", flush=True)
    print(traceback.format_exc(), flush=True)
    sys.exit(1)

# Start the serverless function
print("Starting RunPod serverless handler...", flush=True)
runpod.serverless.start({"handler": handler})
