import torch
from diffusers import DiffusionPipeline
from diffusers.utils import export_to_video

pipeline = DiffusionPipeline.from_pretrained("damo-vilab/text-to-video-ms-1.7b", torch_dtype=torch.float16, variant="fp16")
pipeline = pipeline.to("mps")
# pipeline.enable_model_cpu_offload()
pipeline.enable_vae_slicing()

prompt = "Wide angle shot: A stunning sunset over a futuristic cityscape with towering skyscrapers and flying vehicles. The sky is ablaze with vibrant hues of orange and purple, casting a warm glow on the sleek metallic buildings."
video_frames = pipeline(prompt).frames[0]
export_to_video(video_frames, "modelscopet2v.mp4", fps=10)
