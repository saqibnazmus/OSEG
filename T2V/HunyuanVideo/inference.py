import os
import torch
from diffusers import HunyuanVideoPipeline, HunyuanVideoTransformer3DModel

from diffusers.utils import export_to_video

device = 'cuda'


model_id = "tencent/HunyuanVideo"
transformer = HunyuanVideoTransformer3DModel.from_pretrained(
    model_id, subfolder="transformer", torch_dtype=torch.bfloat16, revision='refs/pr/18'
)





optz = 1
if optz>1:
   from oseg2 import HunyuanVideoOSEGPipeline
   stg_scale = 1.0 # 0.0 for CFG (default)
   guidance_scale =1.3
   
   print('US')
   
elif optz == 1:
   from pipeline_oseg_hunyuan_video import HunyuanVideoOSEGPipeline
   stg_scale = 1.0 # 0.0 for CFG (default)
   guidance_scale = 6
   print('STG')
   
elif optz == 0:
   stg_scale = 0.0 # 0.0 for CFG (default)
   from pipeline_oseg_hunyuan_video import HunyuanVideoOSEGPipeline
   guidance_scale = 6
   print('Baseline')




pipe = HunyuanVideoOSEGPipeline.from_pretrained(model_id, transformer=transformer, revision='refs/pr/18', torch_dtype=torch.float16)
pipe.vae.enable_tiling()
pipe.to(device)

print('o')
#--------Option--------#
oseg_mode = "STG"
oseg_applied_layers_idx = [2]
 
do_rescaling = False
#----------------------#


prompt = "A woman with light skin, wearing a blue jacket and a black hat with a veil, looks down and to her right, then back up as she speaks; she has brown hair styled in an updo, light brown eyebrows, and is wearing a white collared shirt under her jacket; the camera remains stationary on her face as she speaks; the background is out of focus, but shows trees and people in period clothing; the scene is captured in real-life footage."


#prompt = " A butterfly is falpping its wing, close up shot, high resolution"


output = pipe(
    prompt=prompt,
    height=240,
    width=320,
    num_frames=61,
    num_inference_steps=50,
    stg_applied_layers_idx=oseg_applied_layers_idx,
    stg_scale=stg_scale,
    do_rescaling=do_rescaling,
    generator=torch.Generator(device=device).manual_seed(22),
).frames[0]

if optz > 1:  # Check optz > 1 first to prioritize STG_US.mp4
    video_name = "STG_US.mp4"
elif stg_scale == 0:
    video_name = f"CFG_{do_rescaling}.mp4"
elif stg_scale == 1:
    layers_str = "_".join(map(str, oseg_applied_layers_idx))
    video_name = f"{oseg_mode}_rescale_{do_rescaling}.mp4"
# Save video to samples directory
sample_dir = "samples"
os.makedirs(sample_dir, exist_ok=True)
video_path = os.path.join(sample_dir, video_name)
export_to_video(output, video_path, fps=15)

print(f"Video saved to {video_path}")
