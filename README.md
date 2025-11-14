# 🚀[WACV 2026] OSEG: Improving Diffusion sampling through Orthogonal Smoothed Energy Guidance✨

---
## 🏷️ Tags  
![Project](https://img.shields.io/badge/Project-blue.svg)  
![Page](https://img.shields.io/badge/Page-green.svg)  

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
