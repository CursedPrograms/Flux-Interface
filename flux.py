import torch
from diffusers import FluxPipeline
import uuid

# Initialize the model
model_id = "black-forest-labs/FLUX.1-schnell" # You can also use `black-forest-labs/FLUX.1-dev`
pipe = FluxPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
pipe.enable_model_cpu_offload()  # Save some VRAM by offloading the model to CPU. Remove this if you have enough GPU power

# Ask the user for the prompt
prompt = input("Please enter your prompt: ")

# Set the seed and generate the image
seed = 42
image = pipe(
    prompt,
    output_type="pil",
    num_inference_steps=4,  # Use a larger number if you are using [dev]
    generator=torch.Generator("cpu").manual_seed(seed)
).images[0]

# Generate a unique filename using UUID
unique_filename = f"flux-schnell-{uuid.uuid4()}.png"

# Save the generated image with the unique filename
image.save(unique_filename)

print(f"Image saved as {unique_filename}")