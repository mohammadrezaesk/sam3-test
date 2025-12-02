import streamlit as st
import torch
import os
import tempfile
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Check if we should use remote GPU
USE_REMOTE_GPU = os.environ.get("USE_REMOTE_GPU", "").lower() in ("true", "1", "yes")
RUNPOD_ENDPOINT_ID = os.environ.get("RUNPOD_ENDPOINT_ID", "")
RUNPOD_API_KEY = os.environ.get("RUNPOD_API_KEY", "")

# Conditional imports based on GPU mode
if USE_REMOTE_GPU and RUNPOD_ENDPOINT_ID and RUNPOD_API_KEY:
    from runpod_client import RunPodClient, RemoteImageProcessor, should_use_remote_gpu
else:
    # SAM3 local imports
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.visualization_utils import plot_results

# Transformers imports for video (used in both modes)
try:
    from transformers import Sam3VideoModel, Sam3VideoProcessor
    from transformers.video_utils import load_video
    from accelerate import Accelerator
    HAS_VIDEO_SUPPORT = True
except ImportError:
    HAS_VIDEO_SUPPORT = False

# Page config
st.set_page_config(page_title="SAM3 Segmentation", layout="wide")

# Initialize session state
if 'image_model' not in st.session_state:
    st.session_state.image_model = None
if 'image_processor' not in st.session_state:
    st.session_state.image_processor = None
if 'video_model' not in st.session_state:
    st.session_state.video_model = None
if 'video_processor' not in st.session_state:
    st.session_state.video_processor = None
if 'runpod_client' not in st.session_state:
    st.session_state.runpod_client = None
if 'use_remote' not in st.session_state:
    st.session_state.use_remote = USE_REMOTE_GPU and bool(RUNPOD_ENDPOINT_ID) and bool(RUNPOD_API_KEY)


def load_remote_models():
    """Load models from remote RunPod GPU using official SDK"""
    try:
        client = RunPodClient(
            endpoint_id=RUNPOD_ENDPOINT_ID,
            api_key=RUNPOD_API_KEY
        )
        
        # Create remote processor
        confidence_threshold = float(os.environ.get("CONFIDENCE_THRESHOLD", "0.5"))
        remote_processor = RemoteImageProcessor(client, confidence_threshold)
        
        st.session_state.runpod_client = client
        st.session_state.image_processor = remote_processor
        st.session_state.image_model = "remote"  # Marker to indicate remote mode
        st.session_state.video_model = "remote"
        st.session_state.video_processor = client
        
        return True
        
    except Exception as e:
        st.error(f"Error connecting to RunPod: {str(e)}")
        return False


def load_local_models():
    """Load SAM3 models locally"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_path = os.environ.get("SAM3_MODEL_DIR", "sam3-files")
    checkpoint_file = os.path.join(checkpoint_path, "sam3.pt")
    
    if not os.path.exists(checkpoint_file):
        st.error(f"Model checkpoint not found at {checkpoint_file}")
        return False
    
    # Enable TF32 for Ampere GPUs (as per SAM3 examples)
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    try:
        # Load image model
        image_model = build_sam3_image_model(
            checkpoint_path=checkpoint_file,
            load_from_HF=False,
            device=device,
            eval_mode=True
        )
        
        # Convert to float32 to avoid dtype issues
        image_model = image_model.to(device)
        image_model.eval()
        
        confidence_threshold = float(os.environ.get("CONFIDENCE_THRESHOLD", "0.5"))
        image_processor = Sam3Processor(image_model, confidence_threshold=confidence_threshold)
        
        # Load video model and processor using Transformers API
        if HAS_VIDEO_SUPPORT:
            accelerator = Accelerator()
            video_device = accelerator.device
            
            # Use sam3-files directory for from_pretrained (contains config.json)
            model_dir = checkpoint_path
            
            video_model = Sam3VideoModel.from_pretrained(
                model_dir,
                local_files_only=True
            ).to(video_device, dtype=torch.bfloat16 if video_device.type == "cuda" else torch.float32)
            
            video_processor = Sam3VideoProcessor.from_pretrained(
                model_dir,
                local_files_only=True
            )
            
            st.session_state.video_model = video_model
            st.session_state.video_processor = video_processor
        
        st.session_state.image_model = image_model
        st.session_state.image_processor = image_processor
        
        return True
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return False


def load_models():
    """Load SAM3 models (local or remote based on configuration)"""
    if st.session_state.use_remote:
        return load_remote_models()
    else:
        return load_local_models()


def overlay_masks_on_frame(frame, outputs):
    """Overlay masks on a video frame from Transformers API outputs"""
    overlay = frame.copy()
    
    # Handle both dict formats (local and remote)
    masks = outputs.get('masks', [])
    if len(masks) == 0:
        return overlay
    
    object_ids = outputs.get('object_ids', list(range(len(masks))))
    
    # Generate colors
    n_masks = len(masks)
    colors = np.random.RandomState(42).rand(n_masks, 3) * 255
    
    for i, mask in enumerate(masks):
        # Convert mask to numpy if needed
        if hasattr(mask, 'cpu'):
            mask_np = mask.cpu().numpy()
        else:
            mask_np = np.array(mask)
        
        # Ensure mask is 2D and binary
        if mask_np.ndim > 2:
            mask_np = mask_np.squeeze()
        mask_np = (mask_np > 0.5).astype(np.uint8)
        
        # Resize mask to frame size if needed
        if mask_np.shape != frame.shape[:2]:
            mask_np = cv2.resize(mask_np.astype(np.float32), (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST).astype(np.uint8)
        
        # Apply colored overlay
        color = colors[i % len(colors)].astype(np.uint8)
        overlay[mask_np > 0] = overlay[mask_np > 0] * 0.5 + color * 0.5
        
        # Draw bounding box if available
        boxes = outputs.get('boxes', [])
        if i < len(boxes):
            box = boxes[i]
            if hasattr(box, 'cpu'):
                box = box.cpu().numpy()
            box = np.array(box).astype(int)
            cv2.rectangle(overlay, (box[0], box[1]), (box[2], box[3]), color.tolist(), 2)
    
    return overlay


def plot_results_custom(image, output):
    """Custom plot results function that works with both local and remote outputs"""
    import matplotlib.pyplot as plt
    
    masks = output.get("masks", [])
    boxes = output.get("boxes", [])
    scores = output.get("scores", [])
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(image)
    
    # Generate colors
    n_masks = len(masks)
    if n_masks > 0:
        colors = plt.cm.rainbow(np.linspace(0, 1, n_masks))
    
    for i, mask in enumerate(masks):
        # Convert to numpy if needed
        if hasattr(mask, 'cpu'):
            mask_np = mask.cpu().numpy()
        else:
            mask_np = np.array(mask)
        
        # Ensure 2D
        if mask_np.ndim > 2:
            mask_np = mask_np.squeeze()
        
        # Create colored mask overlay
        color = colors[i]
        mask_overlay = np.zeros((*mask_np.shape, 4))
        mask_overlay[mask_np > 0.5] = color
        mask_overlay[..., 3] = mask_np * 0.5  # Semi-transparent
        
        ax.imshow(mask_overlay)
        
        # Draw bounding box
        if i < len(boxes):
            box = boxes[i]
            if hasattr(box, 'cpu'):
                box = box.cpu().numpy()
            box = np.array(box)
            rect = plt.Rectangle(
                (box[0], box[1]), box[2] - box[0], box[3] - box[1],
                linewidth=2, edgecolor=color, facecolor='none'
            )
            ax.add_patch(rect)
            
            # Add score label
            if i < len(scores):
                score = scores[i]
                if hasattr(score, 'item'):
                    score = score.item()
                ax.text(box[0], box[1] - 5, f'{score:.2f}', color=color, fontsize=10)
    
    ax.axis('off')
    return fig


# Sidebar
with st.sidebar:
    st.header("⚙️ Setup")
    
    # GPU Mode selector
    st.subheader("🖥️ GPU Mode")
    
    remote_available = bool(RUNPOD_ENDPOINT_ID) and bool(RUNPOD_API_KEY)
    
    if remote_available:
        gpu_mode = st.radio(
            "Select GPU Mode",
            ["Remote GPU (RunPod)", "Local GPU"],
            index=0 if st.session_state.use_remote else 1,
            help="Choose whether to use remote RunPod GPU or local GPU"
        )
        st.session_state.use_remote = (gpu_mode == "Remote GPU (RunPod)")
        
        if st.session_state.use_remote:
            st.info(f"🌐 Endpoint: {RUNPOD_ENDPOINT_ID[:20]}...")
    else:
        st.info("💻 Using local GPU (RunPod not configured)")
        st.caption("Set RUNPOD_ENDPOINT_ID and RUNPOD_API_KEY to enable remote GPU")
        st.session_state.use_remote = False
    
    st.divider()
    
    if st.button("Load Models", type="primary"):
        with st.spinner("Loading SAM3 models..."):
            if load_models():
                if st.session_state.use_remote:
                    st.success("✅ Connected to RunPod!")
                else:
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                    st.success(f"✅ Models loaded on {device}!")
            else:
                st.error("Failed to load models")
    
    if st.session_state.image_model is not None:
        st.success("Models ready!")
        if st.session_state.use_remote:
            st.info("🌐 Mode: RunPod Serverless")
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            st.info(f"💻 Device: {device}")
        
        if st.session_state.video_model is not None:
            st.info("🎬 Video model ready!")
    
    st.divider()
    
    # Configuration info
    st.subheader("📋 Configuration")
    st.markdown("""
    **Environment Variables:**
    - `USE_REMOTE_GPU`: Enable remote GPU
    - `RUNPOD_ENDPOINT_ID`: Your endpoint ID
    - `RUNPOD_API_KEY`: Your API key
    - `SAM3_MODEL_DIR`: Model directory
    - `CONFIDENCE_THRESHOLD`: Detection threshold
    """)

# Main UI
st.title("🎯 SAM3 Segmentation")
st.markdown("Upload an image or video and enter a text prompt to segment objects")

if st.session_state.image_model is None:
    st.warning("⚠️ Please load the models first using the sidebar.")
    st.stop()

# File upload
uploaded_file = st.file_uploader(
    "Upload Image or Video",
    type=['jpg', 'jpeg', 'png', 'mp4', 'avi', 'mov'],
    help="Upload an image (jpg, png) or video (mp4, avi, mov)"
)

# Prompt input
prompt = st.text_input(
    "Enter segmentation prompt",
    value="person",
    help="Describe what you want to segment (e.g., 'person', 'car', 'dog')"
)

# Process button
process_button = st.button("🚀 Process", type="primary", use_container_width=True)

if process_button and uploaded_file is not None and prompt:
    file_ext = Path(uploaded_file.name).suffix.lower()
    is_video = file_ext in ['.mp4', '.avi', '.mov']
    
    if is_video:
        # Process video
        if st.session_state.video_model is None:
            st.error("Video model not loaded!")
            st.stop()
        
        st.info("Processing video... This may take a while.")
        
        # Save uploaded video to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_video_path = tmp_file.name
        
        try:
            if st.session_state.use_remote:
                # Remote video processing using RunPod SDK
                client = st.session_state.runpod_client
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                status_text.text("Uploading and processing video on RunPod...")
                
                # Process video
                process_result = client.process_video_file(
                    tmp_video_path,
                    prompt,
                    max_frames=30
                )
                
                progress_bar.progress(1.0)
                status_text.text(f"Processed {process_result.num_frames_processed} frames")
                
                # Load video frames for overlay
                video_frames, _ = load_video(tmp_video_path)
                video_frames = video_frames[:process_result.num_frames_processed]
                
                outputs_per_frame = process_result.outputs_per_frame
                
                # Get video properties
                cap = cv2.VideoCapture(tmp_video_path)
                fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                cap.release()
                
            else:
                # Local video processing
                accelerator = Accelerator()
                device = accelerator.device
                
                # Load video frames
                video_frames, _ = load_video(tmp_video_path)
                
                # Limit to first 30 frames
                max_frames_to_process = 30
                video_frames = video_frames[:max_frames_to_process]
                max_frames = len(video_frames)
                
                st.info(f"Processing first {max_frames} frame(s) of the video...")
                
                # Initialize video inference session
                inference_session = st.session_state.video_processor.init_video_session(
                    video=video_frames,
                    inference_device=device,
                    processing_device="cpu",
                    video_storage_device="cpu",
                    dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
                )
                
                # Add text prompt
                inference_session = st.session_state.video_processor.add_text_prompt(
                    inference_session=inference_session,
                    text=prompt,
                )
                
                # Process frames
                progress_bar = st.progress(0)
                status_text = st.empty()
                outputs_per_frame = {}
                
                frame_count = 0
                
                try:
                    for model_outputs in st.session_state.video_model.propagate_in_video_iterator(
                        inference_session=inference_session,
                        max_frame_num_to_track=max_frames
                    ):
                        processed_outputs = st.session_state.video_processor.postprocess_outputs(
                            inference_session, model_outputs
                        )
                        outputs_per_frame[model_outputs.frame_idx] = processed_outputs
                        frame_count += 1
                        
                        progress = frame_count / max_frames
                        progress_bar.progress(min(progress, 1.0))
                        status_text.text(f"Processing frame {model_outputs.frame_idx+1}/{max_frames}")
                except Exception as e:
                    st.warning(f"Processing ended: {str(e)}")
                
                # Get video properties for output
                cap = cv2.VideoCapture(tmp_video_path)
                fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                cap.release()
            
            progress_bar.empty()
            status_text.empty()
            
            st.success(f"✅ Video processed! Processed {len(outputs_per_frame)} frames")
            
            # Create output video with masks
            output_video_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
            
            # Render frames with masks
            for frame_idx in range(len(video_frames)):
                frame = video_frames[frame_idx]
                if isinstance(frame, Image.Image):
                    frame_np = np.array(frame)
                else:
                    frame_np = frame
                
                if frame_idx in outputs_per_frame:
                    frame_outputs = outputs_per_frame[frame_idx]
                    # Overlay masks on frame
                    overlay = overlay_masks_on_frame(frame_np, frame_outputs)
                else:
                    overlay = frame_np
                
                # Write frame
                frame_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
                out.write(frame_bgr)
            
            out.release()
            
            # Display videos
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Original Video")
                st.video(tmp_video_path)
            with col2:
                st.subheader("Processed Video")
                st.video(output_video_path)
            
            # Download button
            with open(output_video_path, "rb") as video_file:
                st.download_button(
                    label="📥 Download Processed Video",
                    data=video_file,
                    file_name=f"segmented_{uploaded_file.name}",
                    mime="video/mp4"
                )
            
            # Show stats
            if len(outputs_per_frame) > 0:
                first_frame_key = min(outputs_per_frame.keys())
                frame_0_outputs = outputs_per_frame[first_frame_key]
                num_objects = len(frame_0_outputs.get('object_ids', frame_0_outputs.get('masks', [])))
                st.info(f"Detected {num_objects} object(s) in first frame")
            
            # Cleanup
            for path in [tmp_video_path, output_video_path]:
                if os.path.exists(path):
                    try:
                        os.unlink(path)
                    except:
                        pass
                        
        except Exception as e:
            st.error(f"Error processing video: {str(e)}")
            st.exception(e)
                        

    
    else:
        # Process image
        with st.spinner("Processing image..."):
            try:
                # Load image
                image = Image.open(uploaded_file).convert("RGB")
                
                # Process using SAM3 (works for both local and remote)
                inference_state = st.session_state.image_processor.set_image(image)
                output = st.session_state.image_processor.set_text_prompt(
                    state=inference_state,
                    prompt=prompt
                )
                
                # Get the masks, bounding boxes, and scores
                masks = output["masks"]
                boxes = output["boxes"]
                scores = output["scores"]
                
                # Display results
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Original Image")
                    st.image(image, use_container_width=True)
                
                with col2:
                    st.subheader("Segmentation Result")
                    # Use custom plot function that works with both local and remote
                    fig = plot_results_custom(image, output)
                    st.pyplot(fig, clear_figure=True)
                
                st.success(f"✅ Found {len(masks)} object(s) matching '{prompt}'")
                
                # Convert scores to list for display
                if hasattr(scores, 'tolist'):
                    scores_list = scores.tolist()
                else:
                    scores_list = list(scores)
                
                if len(scores_list) > 0:
                    st.info(f"Confidence scores: {[f'{s:.3f}' for s in scores_list]}")
                
            except Exception as e:
                st.error(f"Error processing image: {str(e)}")
                st.exception(e)

# Footer
st.markdown("---")
mode_str = "🌐 RunPod Serverless" if st.session_state.use_remote else "💻 Local GPU"
st.markdown(f"**SAM3 (Segment Anything Model 3)** - Promptable segmentation for images and videos | Mode: {mode_str}")
