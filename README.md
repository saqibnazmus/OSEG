# 🚀[WACV 2026] OSEG: Improving Diffusion sampling through Orthogonal Smoothed Energy Guidance✨

---
## 🏷️ Tags  
![Project](https://img.shields.io/badge/Project-blue.svg)  


---

## 🔥 Updates
 
- **2025/11/14** : 🔥 Released the official homepage.

---

## 🚀 Quick Start

### **1. Install Dependencies**
```bash
pip install -r requirements.txt



```markdown
## 🚀 Quickstart (T2I)

Get started with our Jupyter notebook demos:

1. `sdxl_oseg.ipynb`: Basic usage with SDXL  

This is an example of a Python script:

```python
from pipeline_oseg import StableDiffusionXLOSEGPipeline
import torch

pipe = StableDiffusionXLOSEGPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
)

device = "cuda"
pipe = pipe.to(device)

prompts = ["A futuristic cityscape at sunset, ultra-detailed"]
seed = 10

generator = torch.Generator(device=device).manual_seed(seed)

output = pipe(
    prompts,
    num_inference_steps=25,
    guidance_scale=1.0,
    seg_scale=3.0,
    seg_blur_sigma=100.0,
    seg_applied_layers=["mid"],
    generator=generator,
).images

# Save the first image
output[0].save("oseg_example.png")




```markdown
## 🙏 Acknowledgements

This project builds upon the excellent work of:

- [Self-Attention Guidance (SAG)](https://example.com) 
- [Perturbed Attention Guidance (PAG)](https://example.com) by Ahn et al.
- [Smoother Energy Guidance (SEG)](https://example.com) by Ahn et al.


## 🚀 Start Guide

**🧨 Diffusers-based codes**  
To run the test script, refer to the `inference.py` file in each folder.  
Below is an example using **Mochi**:

```python
# inference.py
import torch
from diffusers import MochiPipeline
from pipeline_oseg_mochi import MochiOSEGPipeline
from pipeline_stg_mochi import MochiSTGPipeline
from diffusers.utils import export_to_video
import os

# Ensure the samples directory exists
os.makedirs("samples", exist_ok=True)

ckpt_path = "genmo/mochi-1-preview"
# Load the pipeline
pipe = MochiOSEGPipeline.from_pretrained(
    ckpt_path,
    variant="bf16",
    torch_dtype=torch.bfloat16,
)

# Enable memory savings
# pipe.enable_model_cpu_offload()
# pipe.enable_vae_tiling()
pipe = pipe.to("cuda")

# ------- Options -------
prompt = (
    "A close-up of a beautiful woman's face with colored powder "
    "exploding around her, creating an abstract motion trail"
)
oseg_applied_layers_idx = [34]
oseg_mode = "STG"
stg_scale = 1.0  # 0.0 for CFG (default)
do_rescaling = False  # False (default)
# -----------------------

# Generate video frames
frames = pipe(
    prompt,
    height=480,
    width=480,
    num_frames=81,
    stg_applied_layers_idx=stg_applied_layers_idx,
    stg_scale=stg_scale,
    generator=torch.Generator(device="cuda").manual_seed(42),
    do_rescaling=do_rescaling,
).frames[0]

# Construct the video filename
if stg_scale == 0:
    video_name = f"CFG_rescale_{do_rescaling}.mp4"
else:
    layers_str = "-".join(map(str, oseg_applied_layers_idx))
    video_name = f"{oseg_mode}_scale_{stg_scale}_layers_{layers_str}_rescale_{do_rescaling}.mp4"

# Save video to samples directory
video_path = os.path.join("samples", video_name)
export_to_video(frames, video_path, fps=30)

print(f"Video saved to {video_path}")



```markdown
## 🙏 Acknowledgements

This project builds upon the excellent work of:

- [Mochi](https://example.com)
- [HunyuanVideo](https://example.com) by Apolinário from multimodal AI art  



