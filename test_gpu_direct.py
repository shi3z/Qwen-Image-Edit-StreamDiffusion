#!/usr/bin/env python3
"""
Test GPU-only inference speed with model fully loaded to GPU
"""
import os
# MUST set before importing torch
os.environ['CUDA_VISIBLE_DEVICES'] = '7'

import torch
import gc
from PIL import Image
import numpy as np
import time

def clear_memory():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

def create_test_image(size=(512, 512)):
    img = np.random.randint(0, 255, (*size, 3), dtype=np.uint8)
    return Image.fromarray(img)

def main():
    print("=" * 60)
    print("GPU-Direct Speed Test (Model fully on GPU)")
    print("=" * 60)

    clear_memory()

    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    free_mem = torch.cuda.mem_get_info()[0] / 1e9
    print(f"GPU Free Memory: {free_mem:.1f} GB")

    if free_mem < 50:
        print("ERROR: Need at least 50GB free. Exiting.")
        return

    # Import after env is set
    from diffusers import QwenImageEditPlusPipeline

    print("\n[1] Loading model to GPU...")
    start_time = time.time()

    pipeline = QwenImageEditPlusPipeline.from_pretrained(
        "Qwen/Qwen-Image-Edit-2509",
        torch_dtype=torch.bfloat16,
    ).to('cuda')

    load_time = time.time() - start_time
    print(f"    Loaded in {load_time:.1f}s")
    print(f"    GPU Memory Used: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

    # Test
    test_img = create_test_image((512, 512))
    prompt = "Transform into oil painting style"

    print("\n[2] Warmup...")
    with torch.no_grad():
        _ = pipeline(
            image=[test_img],
            prompt=prompt,
            generator=torch.Generator(device='cuda').manual_seed(42),
            true_cfg_scale=4.0,
            negative_prompt=" ",
            num_inference_steps=4,
            guidance_scale=1.0,
        )
    torch.cuda.synchronize()
    print("    Done")

    print("\n[3] Benchmarking...")
    for steps in [1, 2, 4, 8]:
        times = []
        for i in range(3):
            torch.cuda.synchronize()
            start = time.time()

            with torch.no_grad():
                _ = pipeline(
                    image=[test_img],
                    prompt=prompt,
                    generator=torch.Generator(device='cuda').manual_seed(42 + i),
                    true_cfg_scale=4.0,
                    negative_prompt=" ",
                    num_inference_steps=steps,
                    guidance_scale=1.0,
                )

            torch.cuda.synchronize()
            elapsed = time.time() - start
            times.append(elapsed)

        avg = sum(times) / len(times)
        fps = 1.0 / avg
        print(f"    Steps={steps}: {avg:.2f}s ({fps:.2f} fps), {avg/steps:.3f}s/step")

    print(f"\nMax GPU Memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
    print("Done!")

if __name__ == "__main__":
    main()
