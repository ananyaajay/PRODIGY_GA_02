from diffusers import StableDiffusionPipeline
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
pipe = pipe.to(device)

prompt = "A vibrant futuristic cityscape at sunset, digital art"
result = pipe(prompt)
image = result.images[0]

image.save("output.png")

print("Image saved to output.png")
