# SAM3 Web UI

Simple web interface for SAM3 image and video segmentation.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Make sure you have the model files in `sam3-files/sam3.pt`

## Running the Web UI

Start the Streamlit app:
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## Usage

1. **Load Models**: Click "Load Models" in the sidebar (first time only)
2. **Upload File**: Upload an image (jpg, png) or video (mp4, avi, mov)
3. **Enter Prompt**: Type what you want to segment (e.g., "person", "car", "dog")
4. **Process**: Click the "Process" button
5. **View Results**: See the segmented image/video with colored masks

## Features

- ✅ Image segmentation with visual overlay
- ✅ Video segmentation with mask propagation
- ✅ Download processed results
- ✅ Minimal, clean UI design
- ✅ GPU/CPU automatic detection

## Notes

- First model load may take a minute
- Video processing is limited to 100 frames for demo (can be adjusted)
- Processing time depends on video length and GPU availability

