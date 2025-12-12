#!/usr/bin/env python3
"""
Benchmark Lightning LoRA with different output sizes.
Combines 2-step Lightning LoRA with resolution reduction for maximum speed.
"""

import os
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')

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


def benchmark_size(pipeline, input_size, output_size, num_warmup=1, num_runs=3, num_steps=2):
    """Benchmark at specific input/output size"""
    device = next(pipeline.transformer.parameters()).device

    # Create test image
    test_img = create_test_image(input_size)
    prompt = "Transform this into an oil painting style"

    # Warmup
    print(f"  Warming up...")
    for i in range(num_warmup):
        with torch.no_grad():
            _ = pipeline(
                image=[test_img],
                prompt=prompt,
                generator=torch.Generator(device=device).manual_seed(42),
                true_cfg_scale=4.0,
                negative_prompt=" ",
                num_inference_steps=num_steps,
                guidance_scale=1.0,
                height=output_size[0],
                width=output_size[1],
            )
        torch.cuda.synchronize()

    # Reset peak memory after warmup
    torch.cuda.reset_peak_memory_stats()

    # Benchmark
    times = []
    for i in range(num_runs):
        torch.cuda.synchronize()
        start = time.time()

        with torch.no_grad():
            result = pipeline(
                image=[test_img],
                prompt=prompt,
                generator=torch.Generator(device=device).manual_seed(42 + i),
                true_cfg_scale=4.0,
                negative_prompt=" ",
                num_inference_steps=num_steps,
                guidance_scale=1.0,
                height=output_size[0],
                width=output_size[1],
            )

        torch.cuda.synchronize()
        elapsed = time.time() - start
        times.append(elapsed)

    avg_time = sum(times) / len(times)
    _, _, peak_mem = get_gpu_memory_info()

    return {
        'avg_time': avg_time,
        'min_time': min(times),
        'max_time': max(times),
        'peak_memory': peak_mem,
        'result': result,
    }


def main():
    print("=" * 70)
    print("Lightning LoRA + Resolution Benchmark")
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

    print(f"\n[1] Loading BF16 model with Lightning LoRA...")
    load_start = time.time()

    pipeline = QwenImageEditPlusPipeline.from_pretrained(
        "Qwen/Qwen-Image-Edit-2509",
        torch_dtype=torch.bfloat16
    ).to(device)

    # Load Lightning LoRA for 2-step inference
    print("  Loading 2-step Lightning LoRA...")
    pipeline.load_lora_weights(
        "lightx2v/Qwen-Image-Lightning",
        weight_name="Qwen-Image-Edit-Lightning-4steps-V1.0.safetensors",
    )
    print("  Lightning LoRA loaded!")

    load_time = time.time() - load_start
    alloc, _, _ = get_gpu_memory_info()
    print(f"    Model load time: {load_time:.2f}s")
    print(f"    GPU Memory: {alloc:.2f} GB allocated")

    # Test different output sizes with 2 steps (Lightning LoRA)
    output_sizes = [
        (256, 256),   # Very small
        (384, 384),   # Small
        (512, 512),   # Standard (default)
        (640, 640),   # Medium
        (768, 768),   # Large
        (1024, 1024), # Very large
    ]

    print(f"\n[2] Benchmarking Lightning LoRA (2 steps) at different output sizes...")
    print("-" * 70)
    print(f"{'Output Size':<15} {'Avg Time':<12} {'Min Time':<12} {'Peak Mem':<12} {'FPS':<10}")
    print("-" * 70)

    results = []

    for output_size in output_sizes:
        print(f"\n  Testing {output_size[0]}x{output_size[1]}...")

        try:
            result = benchmark_size(
                pipeline,
                input_size=(512, 512),  # Fixed input
                output_size=output_size,
                num_warmup=1,
                num_runs=3,
                num_steps=2,  # Lightning LoRA 2-step
            )

            fps = 1.0 / result['avg_time']

            results.append({
                'size': output_size,
                **result,
                'fps': fps,
            })

            # Save sample
            result['result'].images[0].save(f"lightning_{output_size[0]}x{output_size[1]}.jpg")

            print(f"  {output_size[0]}x{output_size[1]:<10} {result['avg_time']:.3f}s{'':<6} {result['min_time']:.3f}s{'':<6} {result['peak_memory']:.2f}GB{'':<5} {fps:.3f}")

        except Exception as e:
            print(f"  {output_size[0]}x{output_size[1]}: FAILED - {e}")
            results.append({
                'size': output_size,
                'error': str(e),
            })

        # Clean up between runs
        torch.cuda.empty_cache()
        gc.collect()

    print("-" * 70)

    # Also test 4-step for comparison
    print(f"\n[3] Benchmarking Lightning LoRA (4 steps) at 512x512 for comparison...")
    result_4step = benchmark_size(
        pipeline,
        input_size=(512, 512),
        output_size=(512, 512),
        num_warmup=1,
        num_runs=3,
        num_steps=4,
    )
    print(f"  4-step @ 512x512: {result_4step['avg_time']:.3f}s ({1.0/result_4step['avg_time']:.3f} FPS)")

    # Summary
    print(f"\n[Summary]")
    print("=" * 70)
    print(f"{'Config':<25} {'Time':<12} {'FPS':<10} {'Speedup':<10}")
    print("-" * 70)

    # Find 512x512 2-step baseline
    baseline_time = None
    for r in results:
        if 'error' not in r and r['size'] == (512, 512):
            baseline_time = r['avg_time']
            break

    # Print 4-step reference
    print(f"{'Lightning 4-step 512':<25} {result_4step['avg_time']:.3f}s{'':<6} {1.0/result_4step['avg_time']:.3f}{'':<5} {'(reference)':<10}")

    for r in results:
        if 'error' in r:
            print(f"{'Lightning 2-step ' + str(r['size'][0]):<25} ERROR")
        else:
            speedup = result_4step['avg_time'] / r['avg_time']
            print(f"{'Lightning 2-step ' + str(r['size'][0]):<25} {r['avg_time']:.3f}s{'':<6} {r['fps']:.3f}{'':<5} {speedup:.2f}x")

    print("=" * 70)

    # Best option
    valid_results = [r for r in results if 'error' not in r]
    if valid_results:
        fastest = min(valid_results, key=lambda x: x['avg_time'])
        print(f"\nFastest: Lightning 2-step @ {fastest['size'][0]}x{fastest['size'][1]}")
        print(f"  Time: {fastest['avg_time']:.3f}s")
        print(f"  FPS: {fastest['fps']:.3f}")
        print(f"  Speedup vs 4-step 512: {result_4step['avg_time']/fastest['avg_time']:.2f}x")


if __name__ == "__main__":
    main()
