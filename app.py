import os
import gradio as gr
import torch
from PIL import Image
import numpy as np
import tempfile
from pathlib import Path
import cv2
import matplotlib.pyplot as plt

hf_token = os.getenv("HF_TOKEN")
if hf_token:
    from huggingface_hub import login
    login(token=hf_token)

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.visualization_utils import plot_results

from transformers import Sam3VideoModel, Sam3VideoProcessor
from transformers.video_utils import load_video
from accelerate import Accelerator

device = "cuda" if torch.cuda.is_available() else "cpu"
image_model = None
image_processor = None
video_model = None
video_processor = None

def initialize_models():
    global image_model, image_processor, video_model, video_processor
    if image_model is None:
        image_model = build_sam3_image_model()
        image_processor = Sam3Processor(image_model)
    if video_model is None:
        accelerator = Accelerator()
        video_device = accelerator.device
        video_model = Sam3VideoModel.from_pretrained("facebook/sam3").to(
            video_device, dtype=torch.bfloat16 if video_device.type == "cuda" else torch.float32
        )
        video_processor = Sam3VideoProcessor.from_pretrained("facebook/sam3")

def overlay_masks_on_frame(frame, outputs):
    overlay = frame.copy()
    
    if 'masks' not in outputs or len(outputs['masks']) == 0:
        return overlay
    
    masks = outputs['masks']
    object_ids = outputs.get('object_ids', [])
    
    n_masks = len(masks)
    colors = np.random.RandomState(42).rand(n_masks, 3) * 255
    
    for i, (mask, obj_id) in enumerate(zip(masks, object_ids)):
        if hasattr(mask, 'cpu'):
            mask_np = mask.cpu().numpy()
        else:
            mask_np = np.array(mask)
        
        if mask_np.ndim > 2:
            mask_np = mask_np.squeeze()
        mask_np = (mask_np > 0.5).astype(np.uint8)
        
        if mask_np.shape != frame.shape[:2]:
            mask_np = cv2.resize(
                mask_np.astype(np.float32), 
                (frame.shape[1], frame.shape[0]), 
                interpolation=cv2.INTER_NEAREST
            ).astype(np.uint8)
        
        color = colors[i % len(colors)].astype(np.uint8)
        overlay[mask_np > 0] = overlay[mask_np > 0] * 0.5 + color * 0.5
        
        if 'boxes' in outputs and i < len(outputs['boxes']):
            box = outputs['boxes'][i]
            if hasattr(box, 'cpu'):
                box = box.cpu().numpy()
            box = box.astype(int)
            cv2.rectangle(overlay, (box[0], box[1]), (box[2], box[3]), color.tolist(), 2)
    
    return overlay

def process_image(image, text_prompt, progress=gr.Progress()):
    if image is None or not text_prompt:
        return None, None
    
    progress(0.1, desc="Initializing models...")
    initialize_models()
    
    progress(0.3, desc="Processing image...")
    inference_state = image_processor.set_image(image)
    output = image_processor.set_text_prompt(state=inference_state, prompt=text_prompt)
    
    masks = output["masks"]
    boxes = output["boxes"]
    scores = output["scores"]
    
    progress(0.8, desc="Generating visualization...")
    
    if masks.shape[0] == 0:
        return image, f"No objects found matching '{text_prompt}'"
    
    fig, ax = plt.subplots(figsize=(12, 8))
    plot_results(image, output)
    ax.axis('off')
    
    temp_path = tempfile.mktemp(suffix='.png')
    plt.savefig(temp_path, bbox_inches='tight', pad_inches=0, dpi=100)
    plt.close(fig)
    
    result_image = Image.open(temp_path)
    os.unlink(temp_path)
    
    progress(1.0, desc="Complete!")
    
    return result_image, f"✅ Found {len(masks)} object(s) matching '{text_prompt}'"

def process_video(video_path, text_prompt, max_frames=30, avg_frame=False, progress=gr.Progress()):
    if video_path is None or not text_prompt:
        return None, None
    
    progress(0.1, desc="Initializing models...")
    initialize_models()
    
    file_path = video_path.name if hasattr(video_path, 'name') else video_path
    
    cap = cv2.VideoCapture(file_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    
    progress(0.2, desc="Loading video...")
    video_frames, _ = load_video(file_path)
    
    if avg_frame:
        total_seconds = total_frames // fps
        original_frame_indices = []
        for second in range(total_seconds):
            middle_frame_idx = int((second * fps) + (fps / 2))
            if middle_frame_idx < len(video_frames):
                original_frame_indices.append(middle_frame_idx)
        
        max_frames_to_process = min(len(original_frame_indices), int(max_frames) if max_frames else len(original_frame_indices))
        original_frame_indices = original_frame_indices[:max_frames_to_process]
        selected_frames = [video_frames[idx] for idx in original_frame_indices]
        video_frames = selected_frames
        max_frames = len(video_frames)
    else:
        max_frames_to_process = int(max_frames) if max_frames else 30
        video_frames = video_frames[:max_frames_to_process]
        max_frames = len(video_frames)
        original_frame_indices = list(range(max_frames))
    
    accelerator = Accelerator()
    video_device = accelerator.device
    
    progress(0.3, desc="Initializing video session...")
    inference_session = video_processor.init_video_session(
        video=video_frames,
        inference_device=video_device,
        processing_device="cpu",
        video_storage_device="cpu",
        dtype=torch.bfloat16 if video_device.type == "cuda" else torch.float32,
    )
    
    progress(0.4, desc="Adding text prompt...")
    inference_session = video_processor.add_text_prompt(
        inference_session=inference_session,
        text=text_prompt,
    )
    
    progress(0.5, desc="Processing video frames...")
    outputs_per_frame = {}
    frame_count = 0
    
    try:
        for model_outputs in video_model.propagate_in_video_iterator(
            inference_session=inference_session,
            max_frame_num_to_track=max_frames
        ):
            processed_outputs = video_processor.postprocess_outputs(
                inference_session, model_outputs
            )
            processed_idx = model_outputs.frame_idx
            if processed_idx < len(original_frame_indices):
                original_idx = original_frame_indices[processed_idx]
                outputs_per_frame[original_idx] = processed_outputs
            frame_count += 1
            
            progress_val = 0.5 + (frame_count / max_frames) * 0.4
            progress(progress_val, desc=f"Processing frame {frame_count}/{max_frames}")
    except Exception as e:
        return None, f"Error processing video: {str(e)}"
    
    progress(0.9, desc="Rendering output video...")
    
    output_path = tempfile.mktemp(suffix='.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    all_video_frames, _ = load_video(file_path)
    
    for frame_idx in range(len(all_video_frames)):
        frame = all_video_frames[frame_idx]
        if isinstance(frame, Image.Image):
            frame_np = np.array(frame)
        else:
            frame_np = frame
        
        if frame_idx in outputs_per_frame:
            frame_outputs = outputs_per_frame[frame_idx]
            overlay = overlay_masks_on_frame(frame_np, frame_outputs)
        else:
            overlay = frame_np
        
        frame_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)
    
    out.release()
    
    progress(1.0, desc="Complete!")
    
    num_objects = len(outputs_per_frame.get(0, {}).get('object_ids', [])) if outputs_per_frame else 0
    status_msg = f"✅ Processed {len(outputs_per_frame)} frame(s). Detected {num_objects} object(s)."
    
    return output_path, status_msg

def process_image_input(image_file, text_prompt, progress=gr.Progress()):
    if image_file is None or not text_prompt:
        return None, None, "Please upload an image and enter a prompt."
    
    file_path = image_file.name if hasattr(image_file, 'name') else image_file
    image = Image.open(file_path).convert("RGB")
    processed_image, status = process_image(image, text_prompt, progress)
    return image, processed_image, status

def process_video_input(video_file, text_prompt, max_frames, avg_frame, progress=gr.Progress()):
    if video_file is None or not text_prompt:
        return None, None, "Please upload a video and enter a prompt."
    
    processed_video, status = process_video(video_file, text_prompt, max_frames, avg_frame, progress)
    file_path = video_file.name if hasattr(video_file, 'name') else video_file
    return file_path, processed_video, status

with gr.Blocks() as demo:
    gr.Markdown("# 🎯 SAM3 Segmentation Demo")
    gr.Markdown("Upload an image or video and enter a text prompt to segment objects. The output will show masks and bounding boxes.")
    
    with gr.Tabs():
        with gr.Tab("Image"):
            with gr.Column():
                image_prompt = gr.Textbox(
                    label="Text Prompt",
                    placeholder="Enter text prompt (e.g., 'person', 'car', 'dog')",
                    lines=2,
                    value="person"
                )
                image_file = gr.File(
                    label="Upload Image",
                    file_types=["image"],
                )
                image_submit_btn = gr.Button("🚀 Process", variant="primary", size="lg")
                image_status = gr.Textbox(
                    label="Status",
                    interactive=False,
                    lines=2
                )
                
                with gr.Row():
                    with gr.Column():
                        image_original = gr.Image(
                            label="Original", 
                            type="pil",
                        )
                    with gr.Column():
                        image_result = gr.Image(
                            label="Result", 
                            type="pil",
                        )
        
        with gr.Tab("Video"):
            with gr.Column():
                video_prompt = gr.Textbox(
                    label="Text Prompt",
                    placeholder="Enter text prompt (e.g., 'person', 'car', 'dog')",
                    lines=2,
                    value="person"
                )
                video_file = gr.File(
                    label="Upload Video",
                    file_types=["video"],
                )
                with gr.Row():
                    max_frames_input = gr.Number(
                        label="Max Frames to Process",
                        value=30,
                        minimum=1,
                        maximum=1000,
                        step=1,
                        info="Maximum number of frames to process from the video"
                    )
                    avg_frame_checkbox = gr.Checkbox(
                        label="avg_frame",
                        value=False,
                        info="Process only middle frame of each second (optimized)"
                    )
                video_submit_btn = gr.Button("🚀 Process", variant="primary", size="lg")
                video_status = gr.Textbox(
                    label="Status",
                    interactive=False,
                    lines=2
                )
                
                with gr.Row():
                    with gr.Column():
                        video_original = gr.Video(
                            label="Original",
                        )
                    with gr.Column():
                        video_result = gr.Video(
                            label="Result",
                        )
    
    def update_image_preview(file):
        if file is None:
            return None
        file_path = file.name if hasattr(file, 'name') else file
        return Image.open(file_path).convert("RGB")
    
    def update_video_preview(file):
        if file is None:
            return None
        file_path = file.name if hasattr(file, 'name') else file
        return file_path
    
    image_submit_btn.click(
        fn=process_image_input,
        inputs=[image_file, image_prompt],
        outputs=[image_original, image_result, image_status],
    )
    
    video_submit_btn.click(
        fn=process_video_input,
        inputs=[video_file, video_prompt, max_frames_input, avg_frame_checkbox],
        outputs=[video_original, video_result, video_status],
    )
    
    image_file.change(
        fn=update_image_preview,
        inputs=[image_file],
        outputs=[image_original]
    )
    
    video_file.change(
        fn=update_video_preview,
        inputs=[video_file],
        outputs=[video_original]
    )

if __name__ == "__main__":
    port = int(os.getenv("PORT", "7860"))
    demo.launch(server_name="0.0.0.0", server_port=port)
