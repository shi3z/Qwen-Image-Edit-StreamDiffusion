#!/usr/bin/env python3
"""
Benchmark script for Qwen-Image-Edit-2509 rendering speed
"""

import torch
from diffusers import QwenImageEditPlusPipeline
from PIL import Image
import numpy as np
import time
import gc

def create_test_image(size=(512, 512)):
    """Create a simple test image"""
    img = np.random.randint(0, 255, (*size, 3), dtype=np.uint8)
    return Image.fromarray(img)

def get_gpu_memory_info():
    """Get current GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        max_allocated = torch.cuda.max_memory_allocated() / 1e9
        return allocated, reserved, max_allocated
    return 0, 0, 0

def main():
    print("=" * 70)
    print("Qwen-Image-Edit-2509 Rendering Speed Benchmark")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"\n[Environment]")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU Name: {torch.cuda.get_device_name(0)}")
        props = torch.cuda.get_device_properties(0)
        print(f"  Compute Capability: {props.major}.{props.minor}")
        print(f"  Total Memory: {props.total_memory / 1e9:.2f} GB")

    # Clear cache
    torch.cuda.empty_cache()
    gc.collect()

    print(f"\n[1] Loading model...")
    load_start = time.time()

    pipeline = QwenImageEditPlusPipeline.from_pretrained(
        "Qwen/Qwen-Image-Edit-2509",
        torch_dtype=torch.bfloat16
    ).to(device)

    load_time = time.time() - load_start
    alloc, reserved, max_alloc = get_gpu_memory_info()
    print(f"    Model load time: {load_time:.2f}s")
    print(f"    GPU Memory: {alloc:.2f} GB allocated, {max_alloc:.2f} GB max")

    # Create test image
    test_img = create_test_image((512, 512))
    prompt = "Transform this into an oil painting style"

    print(f"\n[2] Warmup (2 runs)...")
    for i in range(2):
        with torch.no_grad():
            _ = pipeline(
                image=[test_img],
                prompt=prompt,
                generator=torch.Generator(device=device).manual_seed(42),
                true_cfg_scale=4.0,
                negative_prompt=" ",
                num_inference_steps=4,
                guidance_scale=1.0,
            )
        torch.cuda.synchronize()
    print("    Warmup complete")

    # Benchmark different step counts
    print(f"\n[3] Benchmarking (3 runs each)...")
    print("-" * 70)
    print(f"{'Steps':<10} {'Avg Time':<12} {'Min Time':<12} {'Max Time':<12} {'FPS':<10} {'ms/step':<10}")
    print("-" * 70)

    results = []
    for steps in [2, 4, 6, 8]:
        times = []
        for i in range(3):
            torch.cuda.synchronize()
            start = time.time()

            with torch.no_grad():
                _ = pipeline(
                    image=[test_img],
                    prompt=prompt,
                    generator=torch.Generator(device=device).manual_seed(42 + i),
                    true_cfg_scale=4.0,
                    negative_prompt=" ",
                    num_inference_steps=steps,
                    guidance_scale=1.0,
                )

            torch.cuda.synchronize()
            elapsed = time.time() - start
            times.append(elapsed)

        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        fps = 1.0 / avg_time
        ms_per_step = (avg_time / steps) * 1000

        results.append({
            'steps': steps,
            'avg_time': avg_time,
            'min_time': min_time,
            'max_time': max_time,
            'fps': fps,
            'ms_per_step': ms_per_step
        })

        print(f"{steps:<10} {avg_time:.3f}s{'':<6} {min_time:.3f}s{'':<6} {max_time:.3f}s{'':<6} {fps:.3f}{'':<5} {ms_per_step:.1f}ms")

    print("-" * 70)

    # Final memory stats
    alloc, reserved, max_alloc = get_gpu_memory_info()
    print(f"\n[4] Final GPU Memory:")
    print(f"    Allocated: {alloc:.2f} GB")
    print(f"    Max Allocated: {max_alloc:.2f} GB")

    # Summary
    print(f"\n[Summary]")
    print("=" * 70)
    best = min(results, key=lambda x: x['avg_time'])
    print(f"  Fastest config: {best['steps']} steps @ {best['fps']:.3f} FPS ({best['avg_time']:.3f}s/image)")
    print(f"  Recommended for realtime: 2 steps @ {results[0]['fps']:.3f} FPS")
    print("=" * 70)

if __name__ == "__main__":
    main()
