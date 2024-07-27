import torch
import numpy as np
from diffusers import AutoPipelineForImage2Image
from diffusers.utils import load_image, make_image_grid
from PIL import Image

# Force CPU usage
device = torch.device("cpu")
print("Using CPU")

# Enable debugging for diffusers
import logging
logging.basicConfig(level=logging.INFO)

pipeline = AutoPipelineForImage2Image.from_pretrained(
    "kandinsky-community/kandinsky-2-2-decoder",
    torch_dtype=torch.float32,  # Changed to float32 for CPU
    use_safetensors=True
)
pipeline = pipeline.to(device)

# Enable attention slicing for memory efficiency
pipeline.enable_attention_slicing()

# Load and resize the initial image
init_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/cat.png")
init_image = init_image.resize((512, 512))  # Resize to a smaller size

prompt = "cat wizard, gandalf, lord of the rings, detailed, fantasy, cute, adorable, Pixar, Disney, 8k"

try:
    # Generate the image
    output = pipeline(prompt, image=init_image, num_inference_steps=30, guidance_scale=7.5)
    image = output.images[0]

    print(f"Generated image size: {image.size}")
    print(f"Generated image mode: {image.mode}")

    # Check if the image is all black
    if np.array(image).sum() == 0:
        print("Warning: Generated image is all black")

    # Create and save the image grid
    grid = make_image_grid([init_image, image], rows=1, cols=2)
    grid.save("image_grid.png")
    print("Image grid saved successfully.")

    # Save the generated image
    image.save("generated_image.png")
    print("Generated image saved successfully.")

except Exception as e:
    print(f"An error occurred: {e}")

print("Script completed.")