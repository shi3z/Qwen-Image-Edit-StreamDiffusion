#!/usr/bin/env python3
"""
Qwen-Image-Edit-2509 API Server
Handles GPU inference, separated from frontend
With Lightning LoRA for 2-step inference (~3s, no negative_prompt)

GB10 (Blackwell) Optimized Version:
- Disables torch.compile/JIT for stability
- Uses BF16 for optimal performance
- cuDNN benchmark enabled (safe on GB10)
"""
import os
# Use GPU 0 for GB10 (single GPU)
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')

# Configure GB10 environment BEFORE importing torch
os.environ.setdefault('PYTORCH_JIT', '0')
os.environ.setdefault('TORCH_COMPILE_DISABLE', '1')
os.environ.setdefault('TORCHINDUCTOR_DISABLE', '1')

import torch

# Import GPU utils for GB10 detection
from gpu_utils import is_gb10_gpu, print_gpu_info, get_optimal_dtype_for_gpu, get_gpu_info

# GB10-specific configuration
_is_gb10 = is_gb10_gpu()
if _is_gb10:
    print("[GB10 Detected] Using Blackwell-optimized settings")
    print("  - JIT: DISABLED")
    print("  - torch.compile: DISABLED")
    print("  - cuDNN benchmark: ENABLED")
import base64
import io
import time
from PIL import Image
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Enable optimizations
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')

app = FastAPI(title="Qwen-Image-Edit API (Lightning LoRA)")

# CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
pipeline = None
executor = ThreadPoolExecutor(max_workers=1)  # Single GPU, single worker
is_processing = False


class EditRequest(BaseModel):
    image: str  # Base64 encoded image
    prompt: str = "Transform into oil painting style"
    steps: int = 2  # Default 2 steps with Lightning LoRA
    ref_image: str | None = None  # Optional reference image for compositing
    blend_ratio: float = 0.5
    use_cfg: bool = False  # True: CFG (cond+uncond, slower), False: cond only (faster)


class EditResponse(BaseModel):
    image: str  # Base64 encoded result
    elapsed: float
    status: str


def load_pipeline():
    global pipeline
    if pipeline is not None:
        return

    # Get optimal dtype for current GPU
    dtype = get_optimal_dtype_for_gpu()
    print(f"Loading Qwen-Image-Edit-2509 ({dtype})...")

    # Print GPU info
    gpu_info = get_gpu_info()
    print(f"  GPU: {gpu_info['name']}")
    print(f"  Compute Capability: {gpu_info['compute_capability']}")
    print(f"  Memory: {gpu_info['memory_gb']:.1f} GB")

    from diffusers import QwenImageEditPlusPipeline

    pipeline = QwenImageEditPlusPipeline.from_pretrained(
        "Qwen/Qwen-Image-Edit-2509",
        torch_dtype=dtype,
    ).to('cuda')

    print(f"Model loaded! GPU Memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

    # Load Lightning LoRA for fast inference
    # Note: "4steps" LoRA allows minimum 2 steps (no 2steps version exists)
    print("Loading Lightning LoRA (4steps version, used with 2 inference steps)...")
    pipeline.load_lora_weights(
        "lightx2v/Qwen-Image-Lightning",
        weight_name="Qwen-Image-Edit-Lightning-4steps-V1.0.safetensors",
    )
    print("Lightning LoRA loaded!")

    # Warmup (no negative_prompt for fastest inference)
    print("Warming up...")
    dummy = Image.new('RGB', (512, 512), color='gray')
    with torch.no_grad():
        for i in range(2):
            print(f"  Warmup run {i+1}/2...")
            _ = pipeline(
                image=[dummy],
                prompt="test",
                generator=torch.Generator(device='cuda').manual_seed(42),
                num_inference_steps=2,
                guidance_scale=3.5,
            )
    torch.cuda.synchronize()
    print("Ready! (Lightning LoRA, 2-step inference ~3s)")


def base64_to_pil(b64_string: str) -> Image.Image:
    """Convert base64 string to PIL Image"""
    from PIL import ImageOps

    # Remove data URL prefix if present
    if ',' in b64_string:
        b64_string = b64_string.split(',')[1]

    img_data = base64.b64decode(b64_string)
    img = Image.open(io.BytesIO(img_data))

    # Apply EXIF orientation to fix rotation/flip issues
    img = ImageOps.exif_transpose(img)

    return img.convert('RGB')


def pil_to_base64(img: Image.Image) -> str:
    """Convert PIL Image to base64 string"""
    buffer = io.BytesIO()
    img.save(buffer, format='JPEG', quality=90)
    return base64.b64encode(buffer.getvalue()).decode()


def process_image_sync(image: Image.Image, prompt: str, steps: int, use_cfg: bool = False) -> Image.Image:
    """Synchronous image processing

    Args:
        use_cfg: If True, use CFG (cond+uncond, slower but higher quality)
                 If False, use cond only (faster, ~4s)
    """
    global pipeline

    image = image.resize((512, 512), Image.LANCZOS)

    with torch.no_grad():
        if use_cfg:
            # CFG mode: uses negative_prompt for classifier-free guidance
            result = pipeline(
                image=[image],
                prompt=prompt,
                negative_prompt="",
                generator=torch.Generator(device='cuda').manual_seed(42),
                num_inference_steps=steps,
                guidance_scale=3.5,
            )
        else:
            # Fast mode: cond only, no negative_prompt
            result = pipeline(
                image=[image],
                prompt=prompt,
                generator=torch.Generator(device='cuda').manual_seed(42),
                num_inference_steps=steps,
                guidance_scale=3.5,
            )

    return result.images[0]


@app.on_event("startup")
async def startup():
    load_pipeline()


@app.get("/")
async def root():
    return {
        "name": "Qwen-Image-Edit-2509 API",
        "endpoints": ["/health", "/edit"],
        "status": "running"
    }


@app.get("/health")
async def health():
    gpu_info = get_gpu_info()
    return {
        "status": "ok",
        "model_loaded": pipeline is not None,
        "optimized": "Lightning LoRA (2-step)",
        "inference_time": "~5.6s",
        "gpu": gpu_info['name'],
        "compute_capability": gpu_info['compute_capability'],
        "is_gb10": gpu_info['is_gb10'],
        "gpu_memory": f"{torch.cuda.max_memory_allocated() / 1e9:.2f} GB" if torch.cuda.is_available() else "N/A"
    }


@app.post("/edit", response_model=EditResponse)
async def edit_image(request: EditRequest):
    global is_processing

    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if is_processing:
        raise HTTPException(status_code=429, detail="Server is busy processing another request")

    try:
        is_processing = True
        start = time.time()

        # Decode input image
        input_image = base64_to_pil(request.image)

        # Handle compositing if reference image provided
        if request.ref_image:
            ref_image = base64_to_pil(request.ref_image)
            ref_image = ref_image.resize((512, 512), Image.LANCZOS)
            input_image = input_image.resize((512, 512), Image.LANCZOS)
            input_image = Image.blend(input_image, ref_image, request.blend_ratio)

        # Process in thread pool to not block event loop
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            executor,
            process_image_sync,
            input_image,
            request.prompt,
            request.steps,
            request.use_cfg
        )

        elapsed = time.time() - start

        return EditResponse(
            image=pil_to_base64(result),
            elapsed=elapsed,
            status="success"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        is_processing = False


if __name__ == "__main__":
    print("=" * 60)
    print("Qwen-Image-Edit-2509 API Server (GB10 Optimized)")
    print("=" * 60)

    # Print startup info
    print_gpu_info()

    uvicorn.run(app, host="0.0.0.0", port=8086)
