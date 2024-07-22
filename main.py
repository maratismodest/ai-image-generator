from diffusers import StableDiffusionPipeline
import torch

# Load the pipeline
# Use torch.float32 for CPU or MPS (Apple Silicon)
pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float32)

# For Apple Silicon (M1/M2) Macs, uncomment the following line:
pipe = pipe.to("mps")

# For CPU-only operation, you can leave it as is or explicitly set to CPU:
# pipe = pipe.to("cpu")

# Generate an image
prompt = "a cartoon girl like for application"
images = pipe(prompt,num_inference_steps=50).images
# images = pipe(prompt,num_inference_steps=50, num_images_per_prompt=2).images

# Save the image
# image.save("a photo of an guy playing an ukulele.png")

# Save all generated images
for i, image in enumerate(images):
    image.save(f"{prompt}{i}.png")
