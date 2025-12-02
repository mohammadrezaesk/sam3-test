"""
Simple SAM3 Demo Script
This script demonstrates how to use the SAM3 model for video segmentation with text prompts.
"""

import torch
from transformers import Sam3VideoModel, Sam3VideoProcessor
from transformers.video_utils import load_video
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import cv2

def overlay_masks(image, masks, boxes=None, scores=None):
    """Overlay segmentation masks on the image with colors."""
    image = image.convert("RGBA")
    masks_np = masks.cpu().numpy().astype(np.uint8)
    
    n_masks = masks_np.shape[0]
    try:
        cmap = plt.colormaps.get_cmap("rainbow").resampled(n_masks)
    except AttributeError:
        # Fallback for older matplotlib versions
        cmap = plt.cm.get_cmap("rainbow", n_masks)
    colors = [
        tuple(int(c * 255) for c in cmap(i)[:3])
        for i in range(n_masks)
    ]

    for i, (mask, color) in enumerate(zip(masks_np, colors)):
        mask_img = Image.fromarray(mask * 255)
        overlay = Image.new("RGBA", image.size, color + (128,))  # 50% opacity
        alpha = mask_img.point(lambda v: int(v * 0.5))
        overlay.putalpha(alpha)
        image = Image.alpha_composite(image, overlay)
        
        # Optionally draw bounding boxes
        if boxes is not None:
            from PIL import ImageDraw
            draw = ImageDraw.Draw(image)
            box = boxes[i]
            draw.rectangle([box[0], box[1], box[2], box[3]], outline=color, width=2)
            if scores is not None:
                draw.text((box[0], box[1] - 15), f"{scores[i]:.2f}", fill=color)
    
    return image.convert("RGB")

def save_video_with_masks(video_frames, all_results, output_path="sam3_video_result.mp4", fps=30):
    """Save video frames with segmentation masks overlaid."""
    if len(video_frames) == 0:
        print("No frames to save!")
        return
    
    # Get video dimensions from first frame
    first_frame = video_frames[0]
    if isinstance(first_frame, Image.Image):
        height, width = first_frame.size[1], first_frame.size[0]
    else:
        height, width = first_frame.shape[:2]
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print(f"Saving video with {len(video_frames)} frames...")
    for frame_idx, (frame, results) in enumerate(zip(video_frames, all_results)):
        if len(results['masks']) > 0:
            # Convert frame to PIL Image if needed
            if not isinstance(frame, Image.Image):
                frame = Image.fromarray(frame)
            
            # Overlay masks
            overlay_frame = overlay_masks(
                frame,
                results['masks'],
                boxes=results.get('boxes'),
                scores=results.get('scores')
            )
            
            # Convert to numpy array for OpenCV
            frame_np = np.array(overlay_frame)
            frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
        else:
            # No masks, use original frame
            if isinstance(frame, Image.Image):
                frame_np = np.array(frame)
            else:
                frame_np = frame
            frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
        
        out.write(frame_bgr)
        
        if (frame_idx + 1) % 10 == 0:
            print(f"Processed {frame_idx + 1}/{len(video_frames)} frames...")
    
    out.release()
    print(f"Video saved to: {output_path}")

def main():
    print("SAM3 Video Demo - Loading model from local directory...")
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load model from local directory
    model_path = Path("sam3")
    if not model_path.exists():
        print(f"Error: Model directory '{model_path}' not found!")
        return
    
    print(f"Loading model from: {model_path.absolute()}")
    try:
        # Load SAM3 Video Model from local directory
        model = Sam3VideoModel.from_pretrained(str(model_path), local_files_only=True).to(device)
        processor = Sam3VideoProcessor.from_pretrained(str(model_path), local_files_only=True)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("\nTrying alternative loading method...")
        try:
            # Alternative: try without local_files_only
            model = Sam3VideoModel.from_pretrained(str(model_path)).to(device)
            processor = Sam3VideoProcessor.from_pretrained(str(model_path))
            print("Model loaded successfully!")
        except Exception as e2:
            print(f"Error: {e2}")
            print("\nNote: SAM3 requires transformers installed from GitHub main branch:")
            print("  pip install git+https://github.com/huggingface/transformers")
            return
    
    # Load video
    video_path = Path("test.mp4")
    if not video_path.exists():
        print(f"Error: Video file '{video_path}' not found!")
        return
    
    print(f"\nLoading video from: {video_path}")
    try:
        video_frames, _ = load_video(str(video_path))
        print(f"Loaded video with {len(video_frames)} frames")
        print(f"Frame size: {video_frames[0].size if isinstance(video_frames[0], Image.Image) else video_frames[0].shape}")
    except Exception as e:
        print(f"Error loading video: {e}")
        return
    
    # Perform segmentation with text prompt
    text_prompt = "person"  # You can change this to any object you want to segment
    print(f"\nSegmenting objects with text prompt: '{text_prompt}'")
    
    # Process video frames
    print("Processing video frames...")
    all_results = []
    
    # Process frames (you can process all frames or sample them)
    # For demo, let's process all frames but you can add sampling if needed
    max_frames = 30  # Process all frames, or set a limit like 30
    
    for frame_idx in range(min(max_frames, len(video_frames))):
        frame = video_frames[frame_idx]
        
        # Process inputs
        inputs = processor(images=frame, text=text_prompt, return_tensors="pt").to(device)
        
        # Run inference
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Post-process results
        results = processor.post_process_instance_segmentation(
            outputs,
            threshold=0.5,
            mask_threshold=0.5,
            target_sizes=inputs.get("original_sizes").tolist()
        )[0]
        
        all_results.append(results)
        
        num_objects = len(results['masks'])
        if (frame_idx + 1) % 10 == 0 or frame_idx == 0:
            print(f"Frame {frame_idx + 1}/{min(max_frames, len(video_frames))}: Found {num_objects} object(s)")
    
    # Save video with masks
    print("\nSaving output video...")
    save_video_with_masks(video_frames[:max_frames], all_results, "sam3_video_result.mp4")
    
    # Also save a sample frame for preview
    if len(all_results) > 0 and len(all_results[0]['masks']) > 0:
        sample_frame = video_frames[0]
        if not isinstance(sample_frame, Image.Image):
            sample_frame = Image.fromarray(sample_frame)
        
        overlay_image = overlay_masks(
            sample_frame,
            all_results[0]['masks'],
            boxes=all_results[0].get('boxes'),
            scores=all_results[0].get('scores')
        )
        overlay_image.save("sam3_sample_frame.png")
        print("Sample frame saved to: sam3_sample_frame.png")
    
    print("\nDone! Check 'sam3_video_result.mp4' for the segmented video.")

if __name__ == "__main__":
    main()

