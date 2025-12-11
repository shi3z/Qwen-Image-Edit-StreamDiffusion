#!/usr/bin/env python3
"""
Benchmark script optimized for GB10 (NVIDIA Blackwell) GPU
Runs without torch.compile/JIT for maximum compatibility
"""
import os

# Configure environment BEFORE importing torch
os.environ.setdefault('PYTORCH_JIT', '0')
os.environ.setdefault('TORCH_COMPILE_DISABLE', '1')
os.environ.setdefault('TORCHINDUCTOR_DISABLE', '1')

import torch
import time
import gc
from PIL import Image
from diffusers import QwenImageEditPlusPipeline

from gpu_utils import (
    is_gb10_gpu,
    get_optimal_dtype_for_gpu,
    get_gpu_info,
    print_gpu_info,
)


def get_gpu_memory_info():
    """Get current GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        max_allocated = torch.cuda.max_memory_allocated() / 1e9
        return allocated, reserved, max_allocated
    return 0, 0, 0


def benchmark(pipeline, name, steps=4, runs=5, use_lightning=False):
    """Run benchmark and return average time"""
    dummy = Image.new('RGB', (512, 512), color='gray')

    # Warmup
    print(f"  Warming up {name}...")
    with torch.no_grad():
        for _ in range(2):
            if use_lightning:
                # Lightning LoRA mode: no negative_prompt
                _ = pipeline(
                    image=[dummy],
                    prompt="test",
                    generator=torch.Generator(device='cuda').manual_seed(42),
                    num_inference_steps=steps,
                    guidance_scale=3.5,
                )
            else:
                # Standard mode
                _ = pipeline(
                    image=[dummy],
                    prompt="test",
                    generator=torch.Generator(device='cuda').manual_seed(42),
                    true_cfg_scale=4.0,
                    negative_prompt=" ",
                    num_inference_steps=steps,
                    guidance_scale=1.0,
                )
    torch.cuda.synchronize()

    # Benchmark
    times = []
    for i in range(runs):
        torch.cuda.synchronize()
        start = time.time()
        with torch.no_grad():
            if use_lightning:
                _ = pipeline(
                    image=[dummy],
                    prompt="oil painting style",
                    generator=torch.Generator(device='cuda').manual_seed(42),
                    num_inference_steps=steps,
                    guidance_scale=3.5,
                )
            else:
                _ = pipeline(
                    image=[dummy],
                    prompt="oil painting style",
                    generator=torch.Generator(device='cuda').manual_seed(42),
                    true_cfg_scale=4.0,
                    negative_prompt=" ",
                    num_inference_steps=steps,
                    guidance_scale=1.0,
                )
        torch.cuda.synchronize()
        elapsed = time.time() - start
        times.append(elapsed)
        print(f"    Run {i+1}: {elapsed:.3f}s")

    avg = sum(times) / len(times)
    min_t = min(times)
    max_t = max(times)
    fps = 1.0 / avg
    print(f"  {name}: avg={avg:.3f}s, min={min_t:.3f}s, max={max_t:.3f}s, FPS={fps:.2f}")
    return avg, min_t, max_t


def main():
    print("=" * 70)
    print("GB10 (Blackwell) Optimized Benchmark")
    print("Qwen-Image-Edit-2509 Rendering Speed Test")
    print("=" * 70)

    # Check GPU
    print("\n[0] GPU Information")
    print_gpu_info()
    gpu_info = get_gpu_info()

    if gpu_info['is_gb10']:
        print("\n  *** GB10 Detected - Using optimized settings ***")
        print("  - torch.compile: DISABLED")
        print("  - JIT: DISABLED")
        print("  - cuDNN benchmark: ENABLED")
    else:
        print("\n  Note: Not running on GB10, but using GB10-compatible settings")

    # Get optimal dtype
    dtype = get_optimal_dtype_for_gpu()
    print(f"\n  Optimal dtype: {dtype}")

    # Enable cuDNN benchmark (safe on GB10)
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')

    # Clear cache
    torch.cuda.empty_cache()
    gc.collect()

    # Load model
    print(f"\n[1] Loading model ({dtype})...")
    load_start = time.time()

    pipeline = QwenImageEditPlusPipeline.from_pretrained(
        "Qwen/Qwen-Image-Edit-2509",
        torch_dtype=dtype,
    ).to('cuda')

    load_time = time.time() - load_start
    alloc, reserved, max_alloc = get_gpu_memory_info()
    print(f"    Model load time: {load_time:.2f}s")
    print(f"    GPU Memory: {alloc:.2f} GB allocated, {max_alloc:.2f} GB max")

    # Benchmark baseline
    print(f"\n[2] Baseline benchmark (no LoRA)...")
    results = {}

    for steps in [4, 8]:
        avg, min_t, max_t = benchmark(pipeline, f"Baseline {steps} steps", steps=steps, runs=3)
        results[f'baseline_{steps}'] = avg

    # Load Lightning LoRA
    print(f"\n[3] Loading Lightning LoRA...")
    try:
        pipeline.load_lora_weights(
            "lightx2v/Qwen-Image-Lightning",
            weight_name="Qwen-Image-Edit-Lightning-4steps-V1.0.safetensors",
        )
        print("    Lightning LoRA (4-step) loaded!")

        # Benchmark with Lightning LoRA
        print(f"\n[4] Lightning LoRA benchmark...")

        for steps in [2, 4]:
            avg, min_t, max_t = benchmark(
                pipeline,
                f"Lightning {steps} steps",
                steps=steps,
                runs=5,
                use_lightning=True
            )
            results[f'lightning_{steps}'] = avg

    except Exception as e:
        print(f"    Failed to load Lightning LoRA: {e}")
        print("    Continuing with baseline only...")

    # Memory efficiency test
    print(f"\n[5] Memory efficiency test...")
    torch.cuda.reset_peak_memory_stats()

    dummy = Image.new('RGB', (512, 512), color='gray')
    with torch.no_grad():
        _ = pipeline(
            image=[dummy],
            prompt="detailed oil painting",
            generator=torch.Generator(device='cuda').manual_seed(42),
            num_inference_steps=4,
            guidance_scale=3.5,
        )

    alloc, reserved, max_alloc = get_gpu_memory_info()
    print(f"    Peak memory during inference: {max_alloc:.2f} GB")

    # Summary
    print("\n" + "=" * 70)
    print("Summary (GB10 Optimized)")
    print("=" * 70)

    print(f"\nGPU: {gpu_info['name']}")
    print(f"Compute Capability: {gpu_info['compute_capability']}")
    print(f"Memory: {gpu_info['memory_gb']:.2f} GB")
    print(f"Peak Usage: {max_alloc:.2f} GB")

    print(f"\nResults:")
    for name, avg_time in results.items():
        fps = 1.0 / avg_time
        print(f"  {name}: {avg_time:.3f}s ({fps:.2f} FPS)")

    if 'lightning_2' in results and 'baseline_4' in results:
        speedup = results['baseline_4'] / results['lightning_2']
        print(f"\nLightning 2-step vs Baseline 4-step: {speedup:.2f}x faster")

    print("\n" + "=" * 70)
    print("GB10 Optimization Tips:")
    print("- Use Lightning LoRA with 2 steps for fastest inference")
    print("- BF16 dtype provides best performance/quality balance")
    print("- Avoid torch.compile until PyTorch adds Blackwell support")
    print("- cuDNN benchmark is safe to use")
    print("=" * 70)


if __name__ == "__main__":
    main()
