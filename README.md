# SAM3 Video Demo

Simple demo script to test the SAM3 (Segment Anything Model 3) for video segmentation with text prompts.

## Setup

1. **Important**: SAM3 requires transformers from the GitHub main branch (not in released version yet). Install it first:
```bash
pip install git+https://github.com/huggingface/transformers
```

2. Install other dependencies:
```bash
pip install -r requirements.txt
```

**Note**: The transformers installation from GitHub may take a few minutes as it clones the repository.

## Usage

1. Place your video file as `test.mp4` in the project directory
2. Run the demo script:
```bash
python demo_sam3.py
```

The script will:
1. Load the SAM3 video model from the local `sam3` directory
2. Load and process the video file `test.mp4`
3. Perform segmentation on each frame using a text prompt (default: "person")
4. Save the results to `sam3_video_result.mp4` (video with masks overlaid)
5. Save a sample frame to `sam3_sample_frame.png` for preview

## Customization

You can modify the script to:
- Change the text prompt (line with `text_prompt = "person"`)
- Change the video file path (currently `test.mp4`)
- Adjust the number of frames to process (modify `max_frames` variable)
- Adjust segmentation thresholds (currently set to 0.5)

## Model Location

The model files should be in the `sam3/` directory as downloaded from Hugging Face.


