#!/usr/bin/env python3
"""
Benchmark different optimization strategies for Qwen-Image-Edit-2509
GB10 compatible version
"""
import os
# Use GPU 0 for GB10
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')

import torch
import time
from PIL import Image
from diffusers import QwenImageEditPlusPipeline
from gpu_utils import should_use_torch_compile, print_gpu_info, get_optimal_dtype_for_gpu

torch.backends.cudnn.benchmark = True

def benchmark(pipeline, name, steps=4, runs=5):
    """Run benchmark and return average time"""
    dummy = Image.new('RGB', (512, 512), color='gray')

    # Warmup
    print(f"  Warming up {name}...")
    with torch.no_grad():
        for _ in range(2):
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
    print(f"  {name}: avg={avg:.3f}s, min={min(times):.3f}s, max={max(times):.3f}s")
    return avg


def main():
    print("=" * 60)
    print("Qwen-Image-Edit-2509 Optimization Benchmark")
    print("=" * 60)

    # Print GPU info and check compatibility
    print("\n[0] Checking GPU compatibility...")
    print_gpu_info()
    use_compile = should_use_torch_compile()
    print(f"  torch.compile enabled: {use_compile}")

    # Baseline: optimal dtype
    dtype = get_optimal_dtype_for_gpu()
    print(f"\n[1] Loading baseline ({dtype})...")
    pipeline = QwenImageEditPlusPipeline.from_pretrained(
        "Qwen/Qwen-Image-Edit-2509",
        torch_dtype=dtype,
    ).to('cuda')

    baseline = benchmark(pipeline, "Baseline (bf16)")

    # Test xformers
    print("\n[2] Enabling xformers...")
    try:
        pipeline.enable_xformers_memory_efficient_attention()
        xformers_time = benchmark(pipeline, "With xformers")
        print(f"  Speedup: {baseline/xformers_time:.2f}x")
    except Exception as e:
        print(f"  xformers failed: {e}")
        xformers_time = None

    # Test VAE slicing (for memory, might affect speed)
    print("\n[3] Enabling VAE slicing...")
    try:
        pipeline.enable_vae_slicing()
        vae_slice_time = benchmark(pipeline, "With VAE slicing")
        print(f"  Speedup vs baseline: {baseline/vae_slice_time:.2f}x")
    except Exception as e:
        print(f"  VAE slicing failed: {e}")

    # Test torch.compile (if available and supported)
    print("\n[4] Testing torch.compile (may take a while for first compile)...")
    if not use_compile:
        print("  Skipping torch.compile (not supported on this GPU)")
        compile_time = None
    else:
        try:
            # Reset to baseline first
            del pipeline
            torch.cuda.empty_cache()

            pipeline = QwenImageEditPlusPipeline.from_pretrained(
                "Qwen/Qwen-Image-Edit-2509",
                torch_dtype=dtype,
            ).to('cuda')

            # Compile the UNet
            pipeline.transformer = torch.compile(pipeline.transformer, mode="reduce-overhead")
            compile_time = benchmark(pipeline, "With torch.compile", runs=3)
            print(f"  Speedup vs baseline: {baseline/compile_time:.2f}x")
        except Exception as e:
            print(f"  torch.compile failed: {e}")
            compile_time = None

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Baseline (bf16, 4 steps): {baseline:.3f}s")
    if xformers_time:
        print(f"With xformers: {xformers_time:.3f}s ({baseline/xformers_time:.2f}x)")


if __name__ == "__main__":
    main()
