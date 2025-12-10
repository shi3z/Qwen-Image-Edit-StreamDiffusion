# Qwen-Image-Edit-2509 StreamDiffusion WebUI

Fast real-time image editing WebUI using Qwen-Image-Edit-2509 model.
Inspired by StreamDiffusion and StreamDiffusion2

## Speed Improvements

| Configuration | Time/Image | FPS | Speedup |
|--------------|-----------|-----|---------|
| Original (28 steps) | 114.7s | 0.009 fps | 1x |
| Optimized (4 steps) | 9.5s | 0.11 fps | 12x |
| + torch.compile | 6.9s | 0.14 fps | 17x |
| Optimized (2 steps) | 4.1s | 0.24 fps | 28x |

**Achieved 17-28x speedup** with torch.compile optimization

## Architecture

Two deployment options:

### Option 1: Gradio (Simple)
Single-file WebUI, good for single user testing.

### Option 2: Client-Server (Production)
Separated backend API and React frontend, supports multiple clients.

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   React     │────▶│   FastAPI    │────▶│    GPU      │
│  Frontend   │◀────│   Backend    │◀────│   (A100)    │
└─────────────┘     └──────────────┘     └─────────────┘
   Port 3000           Port 8086            CUDA
                    (torch.compile)
```

## Requirements

- NVIDIA GPU (80GB VRAM recommended, e.g., A100)
- Python 3.11+
- CUDA 12.x
- Node.js 18+ (for React frontend)

## Installation

### Backend
```bash
pip install torch diffusers transformers accelerate fastapi uvicorn
```

### Frontend
```bash
cd frontend
npm install
```

## Usage

### Option 1: Gradio (Simple)
```bash
CUDA_VISIBLE_DEVICES=0 python webui_realtime.py
```
Open http://localhost:7865

### Option 2: Client-Server (Production)

1. Start API Server:
```bash
CUDA_VISIBLE_DEVICES=0 python server.py
```

2. Start React Frontend:
```bash
cd frontend
REACT_APP_API_URL=http://your-gpu-server:8086 npm start
```

For webcam access, HTTPS is required:
```bash
ngrok http 3000  # For frontend
```

## Features

- Real-time webcam image editing
- Image upload editing
- Two-image compositing with editing
- Custom prompts
- Multi-client support (with server.py)

## API Endpoints

- `GET /health` - Health check
- `POST /edit` - Edit image
  - `image`: Base64 encoded image
  - `prompt`: Edit instruction
  - `steps`: Inference steps (2-8)
  - `ref_image`: Optional reference image for compositing
  - `blend_ratio`: Blend ratio (0-1)

## Notes

- 1-step inference is numerically unstable (NaN), minimum 2 steps required
- Model size is approximately 67GB (transformer 58GB + VAE 9GB)
- Server queues requests - only one GPU inference at a time
- First startup takes 3-4 minutes for torch.compile JIT compilation
- torch.compile provides ~27% speedup after warmup

## Acknowledgements

This project builds upon the following excellent works:

- **[StreamDiffusionV2](https://streamdiffusionv2.github.io/)** - A streaming system for dynamic and interactive video generation
- **[StreamDiffusion](https://github.com/cumulo-autumn/StreamDiffusion)** - Pipeline-level solution for real-time interactive generation
- **[Qwen-Image-Edit-2509](https://huggingface.co/Qwen/Qwen-Image-Edit-2509)** - Image editing model by Qwen team

## Citation

If you use this project, please cite the following works:

### StreamDiffusionV2
```bibtex
@article{feng2025streamdiffusionv2,
  title={StreamDiffusionV2: A Streaming System for Dynamic and Interactive Video Generation},
  author={Feng, Tianrui and Li, Zhi and Yang, Shuo and Xi, Haocheng and Li, Muyang and Li, Xiuyu and Zhang, Lvmin and Yang, Keting and Peng, Kelly and Han, Song and others},
  journal={arXiv preprint arXiv:2511.07399},
  year={2025}
}
```

### StreamDiffusion
```bibtex
@article{kodaira2023streamdiffusion,
  title={StreamDiffusion: A Pipeline-level Solution for Real-time Interactive Generation},
  author={Akio Kodaira and Chenfeng Xu and Toshiki Hazama and Takanori Yoshimoto and Kohei Ohno and Shogo Mitsuhori and Soichi Sugano and Hanying Cho and Zhijian Liu and Kurt Keutzer},
  year={2023},
  eprint={2312.12491},
  archivePrefix={arXiv},
  primaryClass={cs.CV}
}
```

### Qwen-Image
```bibtex
@misc{wu2025qwenimagetechnicalreport,
  title={Qwen-Image Technical Report},
  author={Wu, Chenfei and Li, Jiahao and Zhou, Jingren and others},
  year={2025},
  eprint={2508.02324},
  archivePrefix={arXiv},
  primaryClass={cs.CV}
}
```

## License

Apache License 2.0
