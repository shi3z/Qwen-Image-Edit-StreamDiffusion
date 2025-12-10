#!/usr/bin/env python3
"""
Test with torch.compile optimization
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '7'

import torch
import gc
from PIL import Image
import numpy as np
import time

def create_test_image(size=(512, 512)):
    img = np.random.randint(0, 255, (*size, 3), dtype=np.uint8)
    return Image.fromarray(img)

def main():
    print("=" * 60)
    print("torch.compile Optimization Test")
    print("=" * 60)

    gc.collect()
    torch.cuda.empty_cache()

    from diffusers import QwenImageEditPlusPipeline

    print("\n[1] Loading model...")
    pipeline = QwenImageEditPlusPipeline.from_pretrained(
        "Qwen/Qwen-Image-Edit-2509",
        torch_dtype=torch.bfloat16,
    ).to('cuda')

    print(f"    GPU Memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

    # Apply torch.compile to transformer
    print("\n[2] Compiling transformer with torch.compile (default mode)...")
    try:
        pipeline.transformer = torch.compile(
            pipeline.transformer,
            mode="default",  # No CUDA graph to avoid issues
            fullgraph=False,
        )
        print("    Compilation configured (will compile on first run)")
    except Exception as e:
        print(f"    Compile failed: {e}")
        return

    test_img = create_test_image((512, 512))
    prompt = "Transform into oil painting style"

    # Warmup with compilation
    print("\n[3] Warmup (triggering compilation)...")
    for i in range(2):
        start = time.time()
        with torch.no_grad():
            _ = pipeline(
                image=[test_img],
                prompt=prompt,
                generator=torch.Generator(device='cuda').manual_seed(42),
                true_cfg_scale=4.0,
                negative_prompt=" ",
                num_inference_steps=2,
                guidance_scale=1.0,
            )
        torch.cuda.synchronize()
        print(f"    Warmup {i+1}: {time.time() - start:.2f}s")

    # Benchmark
    print("\n[4] Benchmarking with compiled model...")
    for steps in [1, 2, 4]:
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
        print(f"    Steps={steps}: {avg:.2f}s ({fps:.2f} fps)")

    print("\nDone!")

if __name__ == "__main__":
    main()
