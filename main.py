# import torch

# if torch.backends.mps.is_available():
#     device = torch.device("mps")
#     x = torch.ones(1, device=device)
#     print(x)
# else:
#     print("MPS device not found.")


# import torch
# from diffusers import StableDiffusionPipeline

# # Check if MPS is available
# mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()

# # Set the device
# device = torch.device("mps" if mps_available else "cpu")
# print(f"Using device: {device}")

# # Load the pipeline
# pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float32)
# pipe = pipe.to(device)

# # Generate an image
# prompt = "a photo of an astronaut riding a horse on mars"
# image = pipe(prompt, num_inference_steps=20).images[0]  

# # Save the image
# image.save("astronaut_on_mars.png")

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
prompt = "a photo of a beautiful Bulgarian woman"
images = pipe(prompt,num_inference_steps=20).images
# images = pipe(prompt,num_inference_steps=50, num_images_per_prompt=2).images

# Save the image
# image.save("a photo of an guy playing an ukulele.png")

# Save all generated images
for i, image in enumerate(images):
    image.save(f"national clothes{i}.png")