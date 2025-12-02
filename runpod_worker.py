"""
SAM3 RunPod Serverless Worker
Deploy this as a serverless endpoint on RunPod.

Usage:
1. Build a Docker image with this file and SAM3 dependencies
2. Deploy to RunPod as a serverless endpoint
3. Use the endpoint ID in your client
"""

import os
import io
import base64
import torch
import numpy as np
from PIL import Image

import runpod

# SAM3 imports
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# Transformers imports for video
from transformers import Sam3VideoModel, Sam3VideoProcessor
from transformers.video_utils import load_video
from accelerate import Accelerator

# Global model instances (loaded once, reused across requests)
image_model = None
image_processor = None
video_model = None
video_processor = None


def load_models():
    """Load SAM3 models on worker startup"""
    global image_model, image_processor, video_model, video_processor
    
    if image_model is not None:
        return  # Already loaded
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading SAM3 models on device: {device}")
    
    # Get model path from environment or use default
    model_dir = os.environ.get("SAM3_MODEL_DIR", "/runpod-volume/sam3-files")
    checkpoint_path = os.path.join(model_dir, "sam3.pt")
    
    if not os.path.exists(checkpoint_path):
        # Try alternative paths
        alt_paths = ["sam3-files/sam3.pt", "/workspace/sam3-files/sam3.pt"]
        for alt in alt_paths:
            if os.path.exists(alt):
                checkpoint_path = alt
                model_dir = os.path.dirname(alt)
                break
        else:
            raise FileNotFoundError(f"Model checkpoint not found. Tried: {checkpoint_path}, {alt_paths}")
    
    # Enable TF32 for Ampere GPUs
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    # Load image model
    print("Loading image model...")
    image_model = build_sam3_image_model(
        checkpoint_path=checkpoint_path,
        load_from_HF=False,
        device=device,
        eval_mode=True
    )
    image_model = image_model.to(device)
    image_model.eval()
    
    confidence_threshold = float(os.environ.get("CONFIDENCE_THRESHOLD", "0.5"))
    image_processor = Sam3Processor(image_model, confidence_threshold=confidence_threshold)
    print("Image model loaded!")
    
    # Load video model
    print("Loading video model...")
    accelerator = Accelerator()
    video_device = accelerator.device
    
    video_model = Sam3VideoModel.from_pretrained(
        model_dir,
        local_files_only=True
    ).to(video_device, dtype=torch.bfloat16 if video_device.type == "cuda" else torch.float32)
    
    video_processor = Sam3VideoProcessor.from_pretrained(
        model_dir,
        local_files_only=True
    )
    print("Video model loaded!")


def decode_base64_image(base64_string: str) -> Image.Image:
    """Decode base64 string to PIL Image"""
    image_data = base64.b64decode(base64_string)
    return Image.open(io.BytesIO(image_data)).convert("RGB")


def encode_image_to_base64(image: Image.Image) -> str:
    """Encode PIL Image to base64 string"""
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def segment_image(image_base64: str, prompt: str, confidence_threshold: float = 0.5):
    """Segment objects in an image using text prompt"""
    # Decode image
    image = decode_base64_image(image_base64)
    
    # Process image
    inference_state = image_processor.set_image(image)
    output = image_processor.set_text_prompt(
        state=inference_state,
        prompt=prompt
    )
    
    # Extract results
    masks = output["masks"]
    boxes = output["boxes"]
    scores = output["scores"]
    
    # Convert masks to list format
    masks_list = []
    for mask in masks:
        if hasattr(mask, 'cpu'):
            mask_np = mask.cpu().numpy()
        else:
            mask_np = np.array(mask)
        masks_list.append(mask_np.astype(int).tolist())
    
    # Convert boxes to list format
    boxes_list = []
    for box in boxes:
        if hasattr(box, 'cpu'):
            box_np = box.cpu().numpy()
        else:
            box_np = np.array(box)
        boxes_list.append(box_np.tolist())
    
    # Convert scores to list format
    if hasattr(scores, 'cpu'):
        scores_list = scores.cpu().numpy().tolist()
    else:
        scores_list = list(scores)
    
    return {
        "masks": masks_list,
        "boxes": boxes_list,
        "scores": scores_list,
        "num_objects": len(masks_list)
    }


def process_video_frames(frames_base64: list, prompt: str, max_frames: int = 30):
    """Process video frames with text prompt"""
    # Decode frames
    frames = [decode_base64_image(f) for f in frames_base64[:max_frames]]
    
    accelerator = Accelerator()
    device = accelerator.device
    
    # Initialize video inference session
    inference_session = video_processor.init_video_session(
        video=frames,
        inference_device=device,
        processing_device="cpu",
        video_storage_device="cpu",
        dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
    )
    
    # Add text prompt
    inference_session = video_processor.add_text_prompt(
        inference_session=inference_session,
        text=prompt,
    )
    
    # Process frames
    outputs_per_frame = {}
    
    for model_outputs in video_model.propagate_in_video_iterator(
        inference_session=inference_session,
        max_frame_num_to_track=max_frames
    ):
        processed_outputs = video_processor.postprocess_outputs(
            inference_session, model_outputs
        )
        
        # Convert outputs to serializable format
        frame_data = {
            "masks": [],
            "boxes": [],
            "object_ids": []
        }
        
        if "masks" in processed_outputs:
            for mask in processed_outputs["masks"]:
                if hasattr(mask, 'cpu'):
                    mask_np = mask.cpu().numpy()
                else:
                    mask_np = np.array(mask)
                frame_data["masks"].append(mask_np.astype(int).tolist())
        
        if "boxes" in processed_outputs:
            for box in processed_outputs["boxes"]:
                if hasattr(box, 'cpu'):
                    box_np = box.cpu().numpy()
                else:
                    box_np = np.array(box)
                frame_data["boxes"].append(box_np.tolist())
        
        if "object_ids" in processed_outputs:
            frame_data["object_ids"] = list(processed_outputs["object_ids"])
        
        outputs_per_frame[model_outputs.frame_idx] = frame_data
    
    return {
        "num_frames_processed": len(outputs_per_frame),
        "outputs_per_frame": outputs_per_frame
    }


def handler(job):
    """
    RunPod serverless handler function.
    
    Expected input format:
    {
        "task": "image" | "video",
        "image_base64": "<base64 string>",  # for image task
        "frames_base64": ["<base64>", ...],  # for video task
        "prompt": "person",
        "confidence_threshold": 0.5,  # optional
        "max_frames": 30  # optional, for video
    }
    """
    job_input = job["input"]
    
    # Ensure models are loaded
    load_models()
    
    task = job_input.get("task", "image")
    prompt = job_input.get("prompt", "")
    
    if not prompt:
        return {"error": "No prompt provided"}
    
    try:
        if task == "image":
            # Image segmentation
            image_base64 = job_input.get("image_base64")
            if not image_base64:
                return {"error": "No image_base64 provided for image task"}
            
            confidence_threshold = job_input.get("confidence_threshold", 0.5)
            result = segment_image(image_base64, prompt, confidence_threshold)
            return result
            
        elif task == "video":
            # Video segmentation
            frames_base64 = job_input.get("frames_base64")
            if not frames_base64:
                return {"error": "No frames_base64 provided for video task"}
            
            max_frames = job_input.get("max_frames", 30)
            result = process_video_frames(frames_base64, prompt, max_frames)
            return result
            
        else:
            return {"error": f"Unknown task: {task}. Use 'image' or 'video'"}
            
    except Exception as e:
        return {"error": str(e)}


# For local testing
if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})

