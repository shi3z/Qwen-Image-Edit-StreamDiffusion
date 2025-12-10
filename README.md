# Qwen-Image-Edit-2509 Fast WebUI

Fast real-time image editing WebUI using Qwen-Image-Edit-2509 model.

## Speed Improvements

| Configuration | Time/Image | FPS |
|--------------|-----------|-----|
| Original (28 steps) | 114.7s | 0.009 fps |
| Optimized (4 steps) | 8.5s | 0.12 fps |
| Optimized (2 steps) | 4.1s | 0.24 fps |

**Achieved 14-28x speedup**

## Requirements

- NVIDIA GPU (80GB VRAM recommended, e.g., A100)
- Python 3.11+
- CUDA 12.x

## Installation

```bash
pip install torch diffusers transformers accelerate gradio
```

## Usage

```bash
CUDA_VISIBLE_DEVICES=0 python webui_realtime.py
```

Open http://localhost:7865 in your browser.

For webcam access, HTTPS is required (use ngrok):
```bash
ngrok http 7865
```

## Features

- Real-time webcam image editing
- Image upload editing
- Two-image compositing with editing
- Custom prompts

## Notes

- 1-step inference is numerically unstable (NaN), minimum 2 steps required
- Model size is approximately 67GB (transformer 58GB + VAE 9GB)
