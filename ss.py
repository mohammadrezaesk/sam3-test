import torch
import os
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Use CUDA if available, otherwise CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    print("Warning: CUDA not available, using CPU (this will be slow)")

def save_image_results(image, masks, boxes, scores, output_dir="outputs"):
    """Save segmentation results for images."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert image to numpy array if needed
    if isinstance(image, Image.Image):
        img_array = np.array(image)
    else:
        img_array = image
    
    # Create overlay visualization
    overlay = img_array.copy()
    
    # Generate colors for each mask
    n_masks = len(masks)
    if n_masks > 0:
        colors = cm.rainbow(np.linspace(0, 1, n_masks))
        
        for i, (mask, color, score) in enumerate(zip(masks, colors, scores)):
            # Convert mask to numpy if it's a tensor
            if hasattr(mask, 'cpu'):
                mask_np = mask.cpu().numpy()
            else:
                mask_np = np.array(mask)
            
            # Ensure mask is 2D and binary
            if mask_np.ndim > 2:
                mask_np = mask_np[0] if mask_np.shape[0] == 1 else mask_np.squeeze()
            mask_np = (mask_np > 0.5).astype(np.uint8)
            
            # Resize mask to image size if needed
            if mask_np.shape != img_array.shape[:2]:
                mask_pil = Image.fromarray(mask_np)
                mask_pil = mask_pil.resize((img_array.shape[1], img_array.shape[0]), Image.NEAREST)
                mask_np = np.array(mask_pil)
            
            # Apply colored overlay
            color_rgb = (np.array(color[:3]) * 255).astype(np.uint8)
            overlay[mask_np > 0] = overlay[mask_np > 0] * 0.5 + color_rgb * 0.5
            
            # Draw bounding box if available
            if boxes is not None and i < len(boxes):
                box = boxes[i]
                if hasattr(box, 'cpu'):
                    box = box.cpu().numpy()
                box = box.astype(int)
                overlay_img = Image.fromarray(overlay)
                draw = ImageDraw.Draw(overlay_img)
                draw.rectangle([box[0], box[1], box[2], box[3]], outline=tuple(color_rgb), width=3)
                draw.text((box[0], box[1] - 20), f"Score: {score:.2f}", fill=tuple(color_rgb))
                overlay = np.array(overlay_img)
    
    # Save overlay image
    overlay_img = Image.fromarray(overlay.astype(np.uint8))
    overlay_path = os.path.join(output_dir, "image_segmentation_result.jpg")
    overlay_img.save(overlay_path)
    print(f"Saved overlay image to: {overlay_path}")
    
    # Save individual masks
    for i, mask in enumerate(masks):
        if hasattr(mask, 'cpu'):
            mask_np = mask.cpu().numpy()
        else:
            mask_np = np.array(mask)
        
        if mask_np.ndim > 2:
            mask_np = mask_np[0] if mask_np.shape[0] == 1 else mask_np.squeeze()
        mask_np = (mask_np > 0.5).astype(np.uint8) * 255
        
        # Resize if needed
        if mask_np.shape != img_array.shape[:2]:
            mask_pil = Image.fromarray(mask_np)
            mask_pil = mask_pil.resize((img_array.shape[1], img_array.shape[0]), Image.NEAREST)
            mask_np = np.array(mask_pil)
        
        mask_img = Image.fromarray(mask_np)
        mask_path = os.path.join(output_dir, f"mask_{i:03d}.png")
        mask_img.save(mask_path)
    
    print(f"Saved {n_masks} individual masks to {output_dir}/")
    
    # Save comparison figure
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    axes[0].imshow(img_array)
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    axes[1].imshow(overlay)
    axes[1].set_title(f"Segmentation Results ({n_masks} objects)")
    axes[1].axis('off')
    
    plt.tight_layout()
    comparison_path = os.path.join(output_dir, "comparison.png")
    plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved comparison image to: {comparison_path}")

#################################### For Image ####################################
from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# Load the model from local checkpoint
checkpoint_path = os.path.join("sam3-files", "sam3.pt")
if not os.path.exists(checkpoint_path):
    raise FileNotFoundError(f"Model checkpoint not found at {checkpoint_path}")

print(f"Loading model from local checkpoint: {checkpoint_path}")
model = build_sam3_image_model(
    checkpoint_path=checkpoint_path,
    load_from_HF=False,  # Don't download from HuggingFace
    device=device,  # Use CUDA if available
    eval_mode=True
)

# Convert all parameters and buffers to float32 to avoid dtype mismatch
def convert_to_float32(module):
    for name, param in module.named_parameters():
        if param.dtype != torch.float32:
            param.data = param.data.float()
    for name, buffer in module.named_buffers():
        if buffer.dtype != torch.float32:
            buffer.data = buffer.data.float()

convert_to_float32(model)
model = model.to(device)
model.eval()
processor = Sam3Processor(model)

# Load an image
if os.path.exists("test.jpg"):
    print("\n=== Processing Image ===")
    image = Image.open("test.jpg")
    inference_state = processor.set_image(image)
    # Prompt the model with text
    output = processor.set_text_prompt(state=inference_state, prompt="person")
    
    # Get the masks, bounding boxes, and scores
    masks, boxes, scores = output["masks"], output["boxes"], output["scores"]
    print(f"Found {len(masks)} object(s) with prompt 'person'")
    if len(masks) > 0:
        print(f"Confidence scores: {[f'{s:.3f}' for s in scores.tolist()]}")
        # Save results
        save_image_results(image, masks, boxes, scores, output_dir="outputs/image")
else:
    print("test.jpg not found, skipping image processing")

#################################### For Video ####################################

from sam3.model_builder import build_sam3_video_predictor

print("Building video predictor...")
video_predictor = build_sam3_video_predictor(
    checkpoint_path=checkpoint_path,
)

if os.path.exists("test.mp4"):
    print("\n=== Processing Video ===")
    video_path = "test.mp4"  # a JPEG folder or an MP4 video file
    print(f"Processing video: {video_path}")
    
    # Start a session
    response = video_predictor.handle_request(
        request=dict(
            type="start_session",
            resource_path=video_path,
        )
    )
    session_id = response["session_id"]
    print(f"Started session: {session_id}")
    
    # Add prompt
    response = video_predictor.handle_request(
        request=dict(
            type="add_prompt",
            session_id=session_id,
            frame_index=0,  # Arbitrary frame index
            text="person",
        )
    )
    output = response["outputs"]
    print(f"Video processing completed! Output keys: {output.keys() if isinstance(output, dict) else 'N/A'}")
    
    # Save video results
    output_dir = "outputs/video"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save output information
    import json
    output_info = {
        "session_id": session_id,
        "video_path": video_path,
        "prompt": "person",
        "frame_index": 0,
    }
    if isinstance(output, dict):
        output_info.update({k: str(type(v)) for k, v in output.items()})
    
    info_path = os.path.join(output_dir, "output_info.json")
    with open(info_path, "w") as f:
        json.dump(output_info, f, indent=2)
    print(f"Saved output info to: {info_path}")
    
    # Try to save masks if available
    if isinstance(output, dict) and "masks" in output:
        masks = output["masks"]
        print(f"Found {len(masks) if hasattr(masks, '__len__') else 'N/A'} mask(s) in output")
        
else:
    print("test.mp4 not found, skipping video processing")

print("\n=== All processing complete! ===")
print("Check the 'outputs' directory for results.")