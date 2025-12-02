import runpod
import requests
from io import BytesIO
from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
# Load the mode
model = build_sam3_image_model()
processor = Sam3Processor(model)

def download_image(url):
    response = requests.get(url)
    image = Image.open(BytesIO(response.content))
    return image

def handler(job):

    job_input = job["input"]
    image = download_image(job_input.get("image_url"))
    prompt = job_input.get("prompt", "person")
    inference_state = processor.set_image(image)
    output = processor.set_text_prompt(state=inference_state, prompt=prompt)
    return output


# Start the serverless function
runpod.serverless.start({"handler": handler})
