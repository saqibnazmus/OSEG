import torch
from diffusers import MochiPipeline
from pipeline_oseg_mochi import MochiOSEGPipeline
from diffusers.utils import export_to_video
import os

# Ensure the samples directory exists
os.makedirs("samples", exist_ok=True)

ckpt_path = "genmo/mochi-1-preview"
# Load the pipeline



optz = 2
if optz>1:
   from pipeline_oseg_mochi import MochiOSEGPipeline
   stg_scale = 1.0 # 0.0 for CFG (default)
   guidance_scale =1.03

elif optz == 1:
   from pipeline_stg_mochi import MochiSTGPipeline
   stg_scale = 1.0 # 0.0 for CFG (default)
   guidance_scale = 6
elif optz == 0:
   stg_scale = 0.0 # 0.0 for CFG (default)
   from pipeline_stg_mochi import MochiSTGPipeline
   guidance_scale = 6




pipe = MochiOSEGPipeline.from_pretrained(ckpt_path, variant="bf16", torch_dtype=torch.bfloat16)

# Enable memory savings
pipe.enable_model_cpu_offload()
pipe.enable_vae_tiling()
pipe = pipe.to("cuda")

#--------Option--------#
prompt = "A close-up of a beautiful woman's face with colored powder exploding around her, creating an abstract splash of vibrant hues, realistic style."


prompt =    "A suited astronaut, with the red dust of Mars clinging to their boots, reaches out to shake hands with an alien being, their skin a shimmering blue, under the pink-tinged sky of the fourth planet. In the background, a sleek silver rocket, a beacon of human ingenuity, stands tall, its engines powered down, as the two representatives of different worlds exchange a historic greeting amidst the desolate beauty of the Martian landscape."
 
prompt = "A close-up of a beautiful woman's face with colored powder exploding around her, creating an abstract splash of vibrant hues, realistic style."


stg_applied_layers_idx = [34]
stg_mode = "STG"

do_rescaling =   False #(default)
#----------------------#

# Generate video frames
frames = pipe(
    prompt, 
    height=480,
    width=480,
    num_frames=40,
    stg_applied_layers_idx=stg_applied_layers_idx,
    stg_scale=stg_scale,
    generator = torch.Generator().manual_seed(42),
    do_rescaling=do_rescaling,
).frames[0]

# Construct the video filename
if optz > 1:  # Check optz > 1 first to prioritize STG_US.mp4
    video_name = "STG_US.mp4"
elif stg_scale == 0:
    video_name = f"CFG_rescale_{do_rescaling}.mp4"
elif stg_scale == 1:
    layers_str = "_".join(map(str, stg_applied_layers_idx))
    video_name = f"{stg_mode}_scale_{stg_scale}_layers_{layers_str}_rescale_{do_rescaling}.mp4"

# Save video to samples directory
video_path = os.path.join("samples", video_name)
export_to_video(frames, video_path, fps=30)

print(f"Video saved to {video_path}")
