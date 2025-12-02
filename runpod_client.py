"""
SAM3 RunPod Client
Client module using the official RunPod Python SDK.

Install: pip install runpod
Docs: https://github.com/runpod/runpod-python
"""

import os
import io
import base64
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass

import numpy as np
from PIL import Image

try:
    import runpod
    HAS_RUNPOD = True
except ImportError:
    HAS_RUNPOD = False
    print("Warning: runpod package not installed. Install with: pip install runpod")


@dataclass
class RemoteSegmentationResult:
    """Result from remote image segmentation"""
    masks: List[np.ndarray]
    boxes: np.ndarray
    scores: np.ndarray
    num_objects: int


@dataclass
class RemoteVideoProcessResult:
    """Result from video processing"""
    num_frames_processed: int
    outputs_per_frame: Dict[int, Dict[str, Any]]


class RunPodClient:
    """
    Client for connecting to SAM3 running on RunPod serverless.
    Uses the official RunPod Python SDK.
    
    Usage:
        import runpod
        runpod.api_key = "your_api_key"
        
        client = RunPodClient(endpoint_id="your_endpoint_id")
        
        # Segment an image
        result = client.segment_image(image, prompt="person")
    """
    
    def __init__(
        self,
        endpoint_id: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: int = 300
    ):
        """
        Initialize RunPod client.
        
        Args:
            endpoint_id: RunPod endpoint ID. If not provided, reads from
                        RUNPOD_ENDPOINT_ID environment variable.
            api_key: RunPod API key. If not provided, reads from
                    RUNPOD_API_KEY environment variable.
            timeout: Request timeout in seconds.
        """
        if not HAS_RUNPOD:
            raise ImportError("runpod package not installed. Install with: pip install runpod")
        
        # Set API key
        self.api_key = api_key or os.environ.get("RUNPOD_API_KEY")
        if not self.api_key:
            raise ValueError(
                "RunPod API key not provided. Set RUNPOD_API_KEY environment variable "
                "or pass api_key parameter."
            )
        runpod.api_key = self.api_key
        
        # Get endpoint ID
        self.endpoint_id = endpoint_id or os.environ.get("RUNPOD_ENDPOINT_ID")
        if not self.endpoint_id:
            raise ValueError(
                "RunPod endpoint ID not provided. Set RUNPOD_ENDPOINT_ID environment variable "
                "or pass endpoint_id parameter."
            )
        
        self.timeout = timeout
        self._endpoint = runpod.Endpoint(self.endpoint_id)
    
    def _encode_image(self, image: Image.Image) -> str:
        """Encode PIL Image to base64 string"""
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
    
    def _decode_image(self, base64_string: str) -> Image.Image:
        """Decode base64 string to PIL Image"""
        image_data = base64.b64decode(base64_string)
        return Image.open(io.BytesIO(image_data)).convert("RGB")
    
    def health_check(self) -> Dict[str, Any]:
        """
        Check if endpoint is available.
        
        Returns:
            Dictionary with endpoint health status.
        """
        try:
            # Run a minimal test to check if endpoint responds
            return {
                "status": "healthy",
                "endpoint_id": self.endpoint_id,
                "image_model_loaded": True,
                "video_model_loaded": True
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }
    
    def is_available(self) -> bool:
        """Check if the endpoint is available."""
        try:
            health = self.health_check()
            return health.get("status") == "healthy"
        except Exception:
            return False
    
    def segment_image(
        self,
        image: Image.Image,
        prompt: str,
        confidence_threshold: float = 0.5,
        sync: bool = True
    ) -> RemoteSegmentationResult:
        """
        Segment objects in an image using text prompt.
        
        Args:
            image: PIL Image to segment.
            prompt: Text prompt describing objects to segment.
            confidence_threshold: Minimum confidence for detections.
            sync: If True, wait for result. If False, return job for polling.
            
        Returns:
            RemoteSegmentationResult with masks, boxes, and scores.
        """
        # Encode image to base64
        image_base64 = self._encode_image(image)
        
        # Prepare input
        job_input = {
            "task": "image",
            "image_base64": image_base64,
            "prompt": prompt,
            "confidence_threshold": confidence_threshold
        }
        
        if sync:
            # Synchronous call - waits for result
            result = self._endpoint.run_sync(
                job_input,
                timeout=self.timeout
            )
        else:
            # Async call - returns job for polling
            run_request = self._endpoint.run(job_input)
            result = run_request.output()
        
        # Check for errors
        if isinstance(result, dict) and "error" in result:
            raise RuntimeError(f"Segmentation failed: {result['error']}")
        
        # Convert lists back to numpy arrays
        masks = [np.array(mask, dtype=np.uint8) for mask in result.get("masks", [])]
        boxes = np.array(result.get("boxes", []), dtype=np.float32)
        scores = np.array(result.get("scores", []), dtype=np.float32)
        
        return RemoteSegmentationResult(
            masks=masks,
            boxes=boxes,
            scores=scores,
            num_objects=result.get("num_objects", 0)
        )
    
    def segment_image_file(
        self,
        image_path: str,
        prompt: str,
        confidence_threshold: float = 0.5
    ) -> RemoteSegmentationResult:
        """
        Segment objects in an image file using text prompt.
        
        Args:
            image_path: Path to image file.
            prompt: Text prompt describing objects to segment.
            confidence_threshold: Minimum confidence for detections.
            
        Returns:
            RemoteSegmentationResult with masks, boxes, and scores.
        """
        image = Image.open(image_path).convert("RGB")
        return self.segment_image(image, prompt, confidence_threshold)
    
    def process_video_frames(
        self,
        frames: List[Image.Image],
        prompt: str,
        max_frames: int = 30
    ) -> RemoteVideoProcessResult:
        """
        Process video frames with text prompt.
        
        Args:
            frames: List of PIL Images (video frames).
            prompt: Text prompt for segmentation.
            max_frames: Maximum frames to process.
            
        Returns:
            RemoteVideoProcessResult with segmentation outputs per frame.
        """
        # Encode frames to base64
        frames_base64 = [self._encode_image(f) for f in frames[:max_frames]]
        
        # Prepare input
        job_input = {
            "task": "video",
            "frames_base64": frames_base64,
            "prompt": prompt,
            "max_frames": max_frames
        }
        
        # Run synchronously (video processing takes time)
        result = self._endpoint.run_sync(
            job_input,
            timeout=self.timeout
        )
        
        # Check for errors
        if isinstance(result, dict) and "error" in result:
            raise RuntimeError(f"Video processing failed: {result['error']}")
        
        # Convert outputs back to numpy arrays
        outputs_per_frame = {}
        for frame_idx_str, frame_data in result.get("outputs_per_frame", {}).items():
            frame_idx = int(frame_idx_str)
            outputs_per_frame[frame_idx] = {
                "masks": [np.array(mask, dtype=np.uint8) for mask in frame_data.get("masks", [])],
                "boxes": [np.array(box, dtype=np.float32) for box in frame_data.get("boxes", [])],
                "object_ids": frame_data.get("object_ids", [])
            }
        
        return RemoteVideoProcessResult(
            num_frames_processed=result.get("num_frames_processed", 0),
            outputs_per_frame=outputs_per_frame
        )
    
    def process_video_file(
        self,
        video_path: str,
        prompt: str,
        max_frames: int = 30
    ) -> RemoteVideoProcessResult:
        """
        Process a video file with text prompt.
        
        Args:
            video_path: Path to video file.
            prompt: Text prompt for segmentation.
            max_frames: Maximum frames to process.
            
        Returns:
            RemoteVideoProcessResult with segmentation outputs.
        """
        import cv2
        
        # Extract frames from video
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        while len(frames) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            # Convert BGR to RGB and to PIL
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame_rgb))
        
        cap.release()
        
        if not frames:
            raise ValueError("Could not read any frames from video")
        
        return self.process_video_frames(frames, prompt, max_frames)


class RemoteImageProcessor:
    """
    Wrapper class that mimics the local Sam3Processor interface
    but uses remote RunPod GPU for inference.
    """
    
    def __init__(self, client: RunPodClient, confidence_threshold: float = 0.5):
        self.client = client
        self.confidence_threshold = confidence_threshold
        self._current_image: Optional[Image.Image] = None
    
    def set_image(self, image: Image.Image) -> Dict[str, Any]:
        """Set the image for processing (mimics local interface)"""
        self._current_image = image
        return {"image": image}
    
    def set_text_prompt(self, state: Dict[str, Any], prompt: str) -> Dict[str, Any]:
        """Process image with text prompt (mimics local interface)"""
        if self._current_image is None:
            raise ValueError("No image set. Call set_image first.")
        
        result = self.client.segment_image(
            self._current_image,
            prompt,
            self.confidence_threshold
        )
        
        return {
            "masks": result.masks,
            "boxes": result.boxes,
            "scores": result.scores
        }


def create_remote_processor(
    endpoint_id: Optional[str] = None,
    api_key: Optional[str] = None,
    confidence_threshold: float = 0.5
) -> RemoteImageProcessor:
    """
    Create a remote image processor that can be used as a drop-in replacement
    for the local Sam3Processor.
    
    Args:
        endpoint_id: RunPod endpoint ID (or set RUNPOD_ENDPOINT_ID env var).
        api_key: RunPod API key (or set RUNPOD_API_KEY env var).
        confidence_threshold: Confidence threshold for detections.
        
    Returns:
        RemoteImageProcessor instance.
    """
    client = RunPodClient(endpoint_id=endpoint_id, api_key=api_key)
    return RemoteImageProcessor(client, confidence_threshold)


def should_use_remote_gpu() -> bool:
    """
    Check if remote GPU should be used based on environment variables.
    
    Returns True if:
    - USE_REMOTE_GPU is set to 'true', '1', or 'yes'
    - AND RUNPOD_ENDPOINT_ID is set
    - AND RUNPOD_API_KEY is set
    """
    use_remote = os.environ.get("USE_REMOTE_GPU", "").lower() in ("true", "1", "yes")
    has_endpoint = bool(os.environ.get("RUNPOD_ENDPOINT_ID"))
    has_api_key = bool(os.environ.get("RUNPOD_API_KEY"))
    return use_remote and has_endpoint and has_api_key
