#!/usr/bin/env python3
"""
Minimal test to measure actual inference speed on GPU
Uses sequential offloading for transformer only (main bottleneck)
"""

import torch
import gc
from diffusers import QwenImageEditPlusPipeline
from PIL import Image
import numpy as np
import time
import os

# Use GPU 7
GPU_ID = 7
os.environ['CUDA_VISIBLE_DEVICES'] = str(GPU_ID)

def clear_memory():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

def create_test_image(size=(512, 512)):
    img = np.random.randint(0, 255, (*size, 3), dtype=np.uint8)
    return Image.fromarray(img)

def main():
    print("=" * 60)
    print("Minimal GPU Speed Test")
    print("=" * 60)

    clear_memory()

    free_mem = torch.cuda.mem_get_info()[0] / 1e9
    print(f"GPU Free Memory: {free_mem:.1f} GB")

    if free_mem < 50:
        print("WARNING: Less than 50GB free. Model may not fit completely.")
        print("Using sequential model CPU offload for memory efficiency.")

    print("\n[1] Loading model...")
    start_time = time.time()

    pipeline = QwenImageEditPlusPipeline.from_pretrained(
        "Qwen/Qwen-Image-Edit-2509",
        torch_dtype=torch.bfloat16,
    )

    # Use sequential CPU offload - only moves one layer at a time
    # This should be faster than enable_model_cpu_offload which moves entire models
    pipeline.enable_sequential_cpu_offload()

    load_time = time.time() - start_time
    print(f"    Loaded in {load_time:.1f}s")

    # Create test
    test_img = create_test_image((512, 512))
    prompt = "Transform this into an oil painting style"

    # Benchmark with different steps
    print("\n[2] Benchmarking...")
    for steps in [2, 4, 8]:
        clear_memory()

        times = []
        for i in range(2):
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
            print(f"    Steps={steps}, run {i+1}: {elapsed:.2f}s")

        avg = sum(times) / len(times)
        print(f"    Steps={steps} avg: {avg:.2f}s, {1/avg:.2f} fps")
        print()

    print("Done!")

if __name__ == "__main__":
    main()
