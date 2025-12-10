#!/usr/bin/env python3
"""
Qwen-Image-Edit-2509 API Server
Handles GPU inference, separated from frontend
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '7'

import torch
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

app = FastAPI(title="Qwen-Image-Edit API")

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
    steps: int = 4
    ref_image: str | None = None  # Optional reference image for compositing
    blend_ratio: float = 0.5


class EditResponse(BaseModel):
    image: str  # Base64 encoded result
    elapsed: float
    status: str


def load_pipeline():
    global pipeline
    if pipeline is not None:
        return

    print("Loading Qwen-Image-Edit-2509...")
    from diffusers import QwenImageEditPlusPipeline

    pipeline = QwenImageEditPlusPipeline.from_pretrained(
        "Qwen/Qwen-Image-Edit-2509",
        torch_dtype=torch.bfloat16,
    ).to('cuda')

    print(f"Model loaded! GPU Memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

    # Warmup
    print("Warming up...")
    dummy = Image.new('RGB', (512, 512), color='gray')
    with torch.no_grad():
        _ = pipeline(
            image=[dummy],
            prompt="test",
            generator=torch.Generator(device='cuda').manual_seed(42),
            true_cfg_scale=4.0,
            negative_prompt=" ",
            num_inference_steps=4,
            guidance_scale=1.0,
        )
    torch.cuda.synchronize()
    print("Ready!")


def base64_to_pil(b64_string: str) -> Image.Image:
    """Convert base64 string to PIL Image"""
    # Remove data URL prefix if present
    if ',' in b64_string:
        b64_string = b64_string.split(',')[1]

    img_data = base64.b64decode(b64_string)
    return Image.open(io.BytesIO(img_data)).convert('RGB')


def pil_to_base64(img: Image.Image) -> str:
    """Convert PIL Image to base64 string"""
    buffer = io.BytesIO()
    img.save(buffer, format='JPEG', quality=90)
    return base64.b64encode(buffer.getvalue()).decode()


def process_image_sync(image: Image.Image, prompt: str, steps: int) -> Image.Image:
    """Synchronous image processing"""
    global pipeline

    image = image.resize((512, 512), Image.LANCZOS)

    with torch.no_grad():
        result = pipeline(
            image=[image],
            prompt=prompt,
            generator=torch.Generator(device='cuda').manual_seed(42),
            true_cfg_scale=4.0,
            negative_prompt=" ",
            num_inference_steps=steps,
            guidance_scale=1.0,
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
    return {
        "status": "ok",
        "model_loaded": pipeline is not None,
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
            request.steps
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
    print("Qwen-Image-Edit-2509 API Server")
    print("=" * 60)

    uvicorn.run(app, host="0.0.0.0", port=8086)
