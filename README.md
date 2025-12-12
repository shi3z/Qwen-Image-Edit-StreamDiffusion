# Qwen-Image-Edit-2509 StreamDiffusion WebUI

Fast real-time image editing WebUI using Qwen-Image-Edit-2509 model.
Inspired by StreamDiffusion and StreamDiffusion2

Work in Progress

## Speed Improvements

| Configuration | Time/Image | FPS | Speedup |
|--------------|-----------|-----|---------|
| Original (28 steps) | 114.7s | 0.009 fps | 1x |
| Optimized (4 steps) | 9.5s | 0.11 fps | 12x |
| + torch.compile | 6.9s | 0.14 fps | 17x |
| **Lightning LoRA (2 steps)** | **5.6s** | **0.18 fps** | **20x** |

**Achieved 20x speedup** with Lightning LoRA optimization

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
- Lightning LoRA (lightx2v/Qwen-Image-Lightning) provides stable 5.6s inference
- torch.compile is incompatible with LoRA (causes recompilation issues)

## Python Files

### Core Application

| File | Description |
|------|-------------|
| `server.py` | FastAPI backend server with Lightning LoRA. Handles GPU inference, CORS, request queuing |
| `webui_realtime.py` | Gradio WebUI for single-user webcam/image editing |
| `qwen_realtime.py` | StreamDiffusion2-style acceleration pipeline with latent caching and temporal consistency |

### Pipeline Implementations

| File | Description |
|------|-------------|
| `cached_pipeline.py` | Cached pipeline - reuses prompt/image embeddings for repeated inference |
| `cached_pipeline_v2.py` | Improved cached pipeline with VLM (Vision Language Model) cache support |
| `batched_cfg_pipeline.py` | Batched CFG - combines cond/uncond passes into single batch for ~1.5-1.7x speedup |
| `parallel_cfg_pipeline.py` | Parallel CFG using 2 GPUs (GPU6: cond, GPU7: uncond) with CUDA streams |
| `parallel_cfg_v2.py` ~ `parallel_cfg_v8.py` | Iterative improvements to parallel CFG implementation |
| `parallel_cfg_int8.py` | Parallel CFG with INT8 quantization |
| `parallel_cfg_int8_v2.py` | Improved parallel CFG with INT8 |

### Quantization

| File | Description |
|------|-------------|
| `int8_linear.py` | INT8 Linear layer using cuBLAS Lt GEMM. ~50% memory reduction |
| `int8_memory_optimized.py` | Memory-optimized INT8 implementation |
| `quantize_transformer.py` | Script to replace nn.Linear with Int8Linear in transformer |
| `cublaslt_int8.py` | INT8 GEMM using cuBLAS Lt via CuPy for Tensor Core acceleration |
| `triton_int8_gemm.py` | Triton kernel for fused INT8 GEMM (quantize + matmul + dequantize) |
| `triton_int8_gemm_v2.py` | Improved Triton INT8 GEMM kernel |

### Benchmarks

| File | Description |
|------|-------------|
| `benchmark_lightning.py` | Benchmark Lightning LoRA for faster inference |
| `benchmark_lightning_compile.py` | Benchmark Lightning LoRA with torch.compile |
| `benchmark_compile.py` | Benchmark torch.compile with max-autotune mode |
| `benchmark_optimizations.py` | General optimization benchmarks |
| `benchmark_bnb.py` | Benchmark BitsAndBytes NF4 quantization |
| `benchmark_bnb_int8.py` | Benchmark BitsAndBytes INT8 quantization |
| `benchmark_bnb_int8_v2.py` | Improved BitsAndBytes INT8 benchmark |
| `benchmark_nunchaku.py` | Benchmark Nunchaku INT4 quantization |
| `benchmark_cached.py` | Benchmark cached pipeline performance |
| `benchmark_vision_cache.py` | Benchmark VLM cache effectiveness |
| `benchmark_batched_cfg.py` | Benchmark batched CFG pipeline |
| `benchmark_batched_cfg_impl.py` | Batched CFG implementation benchmark |
| `benchmark_parallel_cfg.py` | Benchmark parallel CFG (2-GPU) |
| `benchmark_parallel_cfg_v2.py` ~ `benchmark_parallel_cfg_v6.py` | Parallel CFG benchmark variants |
| `benchmark_parallel_cfg_e2e.py` | End-to-end parallel CFG benchmark |
| `benchmark_parallel_cfg_int8.py` | Parallel CFG with INT8 benchmark |
| `benchmark_parallel_simple.py` | Simplified parallel benchmark |
| `benchmark_int8.py` | INT8 quantization benchmark |
| `benchmark_int8_v2.py` | Improved INT8 benchmark |
| `benchmark_int8_quantization.py` | INT8 quantization accuracy test |
| `benchmark_int8_only.py` | INT8-only inference benchmark |
| `benchmark_int8_speed.py` | INT8 speed comparison |
| `benchmark_torch_int_mm.py` | Benchmark torch._int_mm for INT8 matmul |

### Utilities & Tests

| File | Description |
|------|-------------|
| `compare_cfg_quality.py` | Compare image quality: True CFG vs No CFG (side-by-side) |
| `test_qwen_edit.py` | Basic Qwen-Image-Edit model test |
| `test_gpu_speed.py` | GPU speed test |
| `test_gpu_direct.py` | Direct GPU access test |
| `test_minimal.py` | Minimal inference test |
| `test_quantized.py` | Quantized model test |
| `test_compiled.py` | torch.compile test |
| `test_vlm_cache.py` | VLM cache functionality test |

## Acknowledgements

This project builds upon the following excellent works:

- **[StreamDiffusionV2](https://streamdiffusionv2.github.io/)** - A streaming system for dynamic and interactive video generation
- **[StreamDiffusion](https://github.com/cumulo-autumn/StreamDiffusion)** - Pipeline-level solution for real-time interactive generation
- **[Qwen-Image-Edit-2509](https://huggingface.co/Qwen/Qwen-Image-Edit-2509)** - Image editing model by Qwen team
- **[Qwen-Image-Lightning](https://huggingface.co/lightx2v/Qwen-Image-Lightning)** - Lightning LoRA for fast 2-step inference

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

## Experimental Results Log

### 2025-12-11: Custom FP8 Quantization Experiment (Failed)

**Goal**: Implement custom FP8 inference pipeline for the transformer to reduce memory and improve speed.

**Environment**:
- GPU: NVIDIA GB10 (Blackwell, Compute Capability 12.1)
- VRAM: 128.46 GB
- PyTorch: 2.8+ with CUDA 12.x
- Note: GB10 is outside PyTorch's officially supported range (CC 8.0-12.0)

**Implementation**:
- Created `fp8_linear.py` with two implementations:
  - `FP8LinearSlow`: Manual dequantize → matmul → result (validation mode)
  - `FP8LinearFast`: Uses `torch._scaled_mm` for native FP8 GEMM
- Weights quantized to FP8 e4m3 (4 exponent, 3 mantissa bits, max=448.0)
- Activations quantized to FP8 e5m2 (5 exponent, 2 mantissa bits, max=57344.0)
- Per-channel scaling for weights, per-tensor scaling for activations
- Calibration via `fp8_calibrate.py` (8 samples, 2 steps each)

**Results** (4 steps, 512x512):

| Configuration | Avg Time | vs BF16 | Peak Memory |
|--------------|----------|---------|-------------|
| **BF16 Baseline** | **28.809s** | **1.00x** | **57.79 GB** |
| FP8 Slow | 44.597s | 0.65x (35% slower) | 78.64 GB |
| FP8 Fast | 63.797s | 0.45x (55% slower) | 78.88 GB |

**Findings**:
1. `torch._scaled_mm` works on GB10 after fixing matrix dimensions (must be 16-divisible) and layout (row-major × column-major for cuBLASLt)
2. Both FP8 modes are **slower** than BF16 baseline
3. FP8 Fast (native `_scaled_mm`) is paradoxically slower than FP8 Slow (manual dequantize)
4. Memory usage is **higher** with FP8 due to storing scales and intermediate tensors

**Analysis**:
- GB10 (Blackwell) FP8 GEMM kernels in PyTorch may not be optimized for CC 12.1
- PyTorch's `_scaled_mm` requires specific matrix layouts that add overhead
- The quantization/dequantization overhead outweighs any FP8 compute benefits
- Native BF16 is well-optimized on this hardware

**Conclusion**: Custom FP8 quantization is **not beneficial** on GB10 with current PyTorch. Recommend using:
- Native BF16 (fastest at 28.8s)
- Lightning LoRA for 2-step inference (5.6s)
- torchao's FP8 if needed (may have better optimization)

**Files Created**:
- `fp8_linear.py` - FP8 Linear layer implementations
- `fp8_calibrate.py` - Calibration script for activation statistics
- `fp8_pipeline.py` - FP8 pipeline integration and benchmark
- `fp8_checkpoint/` - Calibration data (weight scales, activation scales)

---

### 2025-12-12: Output Resolution Benchmark (BF16)

**Goal**: Test if smaller output resolution improves inference speed more effectively than FP8 quantization.

**Environment**: Same as above (GB10, 128.46GB VRAM)

**Results** (BF16, 4 steps, 3 runs each):

| Output Size | Avg Time | vs 512x512 | Peak Memory | Pixels/sec |
|-------------|----------|------------|-------------|------------|
| **256x256** | **15.34s** | **1.20x faster** | 62.27 GB | 4.3k |
| 384x384 | 17.29s | 1.07x faster | 62.27 GB | 8.5k |
| 512x512 | 18.43s | 1.00x (baseline) | 62.27 GB | 14.2k |
| 640x640 | 20.69s | 0.89x | 62.27 GB | 19.8k |
| 768x768 | 23.22s | 0.79x | 62.88 GB | 25.4k |
| 1024x1024 | 29.36s | 0.63x | 66.85 GB | 35.7k |

**Key Findings**:
1. **Resolution scaling is sublinear**: 4x more pixels (256→512) only adds ~20% time
2. **256x256 is fastest**: 15.34s (vs FP8's 44s - **2.9x faster**)
3. **Memory nearly constant**: 62-67GB regardless of output size
4. **Higher resolution = better throughput**: 1024x1024 produces 35.7k pixels/sec vs 4.3k for 256x256

**Conclusion**:
- **Reducing output resolution is far more effective than FP8 quantization**
- For real-time use: 256x256 @ 15.34s beats FP8 512x512 @ 44s
- For quality: 1024x1024 only costs 29s (same as FP8 512x512 baseline)

**Files Created**:
- `benchmark_bf16_sizes.py` - Resolution benchmark script

---

### 2025-12-12: Lightning LoRA + Resolution Benchmark

**Goal**: Combine Lightning LoRA (2-step inference) with output resolution reduction for maximum speed.

**Environment**: Same as above (GB10, 128.46GB VRAM)

**Results** (Lightning LoRA, 2 steps, 3 runs each):

| Config | Avg Time | FPS | Speedup vs 4-step 512 |
|--------|----------|-----|----------------------|
| Lightning 4-step 512 | 22.961s | 0.044 | (reference) |
| **Lightning 2-step 256** | **10.245s** | **0.098** | **2.24x** |
| Lightning 2-step 384 | 11.058s | 0.090 | 2.08x |
| Lightning 2-step 512 | 11.911s | 0.084 | 1.93x |
| Lightning 2-step 640 | 13.317s | 0.075 | 1.72x |
| Lightning 2-step 768 | 14.961s | 0.067 | 1.53x |
| Lightning 2-step 1024 | 19.172s | 0.052 | 1.20x |

**Key Findings**:
1. **Lightning 2-step + 256x256 = 10.245s** (fastest combination)
2. **Resolution reduction still helps**: 256x256 is 16% faster than 512x512 with Lightning LoRA
3. **2-step vs 4-step**: 2-step is consistently ~1.9-2.2x faster at same resolution
4. **Memory constant**: 63-68GB regardless of resolution

**Comparison with BF16 4-step**:
| Method | 512x512 Time | 256x256 Time |
|--------|--------------|--------------|
| BF16 4-step | 18.43s | 15.34s |
| Lightning 2-step | 11.91s | 10.25s |
| **Speedup** | **1.55x** | **1.50x** |

**Conclusion**:
- **Lightning 2-step @ 256x256 is the fastest option** at 10.245s per image
- Combining step reduction and resolution reduction provides cumulative benefits
- For maximum quality: Lightning 2-step @ 512x512 (11.9s) provides good balance
- For maximum speed: Lightning 2-step @ 256x256 (10.2s) is optimal

**Files Created**:
- `benchmark_lightning_sizes.py` - Lightning LoRA + resolution benchmark script

---

## License

Apache License 2.0
