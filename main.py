import torch
from diffusers import FluxPipeline

model_id = "black-forest-labs/FLUX.1-schnell" #you can also use `black-forest-labs/FLUX.1-dev`

pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16)
pipe.enable_model_cpu_offload() #save some VRAM by offloading the model to CPU. Remove this if you have enough GPU power

prompt = "A cat holding a sign that says hello world"
seed = 42
image = pipe(
    prompt,
    output_type="pil",
    num_inference_steps=4, #use a larger number if you are using [dev]
    generator=torch.Generator("cpu").manual_seed(seed)
).images[0]
image.save("flux-schnell.png")