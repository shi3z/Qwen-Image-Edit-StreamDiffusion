#!/usr/bin/env python3
"""
Test GPU-only inference speed (no CPU offloading)
Requires ~47GB GPU memory
"""

import torch
from diffusers import QwenImageEditPlusPipeline
from PIL import Image
import numpy as np
import time
import os

# Use GPU 7 which has most free memory (~48GB)
GPU_ID = 7
os.environ['CUDA_VISIBLE_DEVICES'] = str(GPU_ID)

def create_test_image(size=(512, 512)):
    """Create a simple test image"""
    img = np.random.randint(0, 255, (*size, 3), dtype=np.uint8)
    return Image.fromarray(img)

def main():
    print("=" * 60)
    print("Qwen-Image-Edit-2509 GPU-Only Speed Test")
    print("=" * 60)

    device = torch.device('cuda:0')  # Will be GPU 7 due to CUDA_VISIBLE_DEVICES

    print(f"\nCUDA available: {torch.cuda.is_available()}")
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory Total: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Clear cache
    torch.cuda.empty_cache()

    print("\n[1] Loading model directly to GPU (no CPU offloading)...")
    start_time = time.time()

    try:
        pipeline = QwenImageEditPlusPipeline.from_pretrained(
            "Qwen/Qwen-Image-Edit-2509",
            torch_dtype=torch.bfloat16
        ).to(device)

        load_time = time.time() - start_time
        print(f"    Loaded in {load_time:.1f}s")
        print(f"    GPU Memory Used: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

    except torch.cuda.OutOfMemoryError as e:
        print(f"    CUDA OOM: {e}")
        print("    Falling back to CPU offloading...")

        torch.cuda.empty_cache()
        pipeline = QwenImageEditPlusPipeline.from_pretrained(
            "Qwen/Qwen-Image-Edit-2509",
            torch_dtype=torch.bfloat16
        )
        pipeline.enable_model_cpu_offload()
        print("    Using CPU offloading (will be slower)")

    # Create test image
    test_img = create_test_image((512, 512))
    prompt = "Transform this into an oil painting style"

    print("\n[2] Warmup inference...")
    with torch.no_grad():
        _ = pipeline(
            image=[test_img],
            prompt=prompt,
            generator=torch.manual_seed(42),
            true_cfg_scale=4.0,
            negative_prompt=" ",
            num_inference_steps=4,
            guidance_scale=1.0,
        )
    torch.cuda.synchronize()
    print("    Warmup complete")

    # Benchmark different step counts
    print("\n[3] Benchmarking inference speed...")
    for steps in [1, 2, 4, 8]:
        times = []
        for i in range(3):
            torch.cuda.synchronize()
            start = time.time()

            with torch.no_grad():
                _ = pipeline(
                    image=[test_img],
                    prompt=prompt,
                    generator=torch.manual_seed(42 + i),
                    true_cfg_scale=4.0,
                    negative_prompt=" ",
                    num_inference_steps=steps,
                    guidance_scale=1.0,
                )

            torch.cuda.synchronize()
            elapsed = time.time() - start
            times.append(elapsed)

        avg_time = sum(times) / len(times)
        fps = 1.0 / avg_time
        time_per_step = avg_time / steps

        print(f"    Steps={steps}: {avg_time:.2f}s total, {time_per_step:.3f}s/step, {fps:.2f} fps")

    print("\n[4] Final GPU Memory:")
    print(f"    Max allocated: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
    print(f"    Current allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    print("\n" + "=" * 60)
    print("Test completed!")
    print("=" * 60)

if __name__ == "__main__":
    main()
