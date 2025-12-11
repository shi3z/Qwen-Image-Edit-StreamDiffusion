#!/usr/bin/env python3
"""
Benchmark torch.compile with max-autotune mode for Qwen-Image-Edit-2509
This uses CUDA graphs and kernel fusion without needing TensorRT

Note: torch.compile is automatically skipped on GB10 (Blackwell) GPUs
"""
import os
# Use GPU 0 for GB10
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')

import torch
import time
from PIL import Image
from diffusers import QwenImageEditPlusPipeline
from gpu_utils import should_use_torch_compile, print_gpu_info, safe_torch_compile, get_optimal_dtype_for_gpu

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')

def benchmark(pipeline, name, steps=4, runs=5):
    """Run benchmark and return average time"""
    dummy = Image.new('RGB', (512, 512), color='gray')

    # Warmup (more runs for compiled model)
    print(f"  Warming up {name}...")
    with torch.no_grad():
        for _ in range(3):
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
    print("torch.compile Optimization Benchmark")
    print("=" * 60)

    # Print GPU info and check compatibility
    print("\n[0] Checking GPU compatibility...")
    print_gpu_info()
    use_compile = should_use_torch_compile()
    print(f"  torch.compile enabled: {use_compile}")

    # Load model
    dtype = get_optimal_dtype_for_gpu()
    print(f"\n[1] Loading model ({dtype})...")
    pipeline = QwenImageEditPlusPipeline.from_pretrained(
        "Qwen/Qwen-Image-Edit-2509",
        torch_dtype=dtype,
    ).to('cuda')

    mem = torch.cuda.max_memory_allocated() / 1e9
    print(f"  GPU Memory: {mem:.2f} GB")

    # Baseline
    print("\n[2] Baseline benchmark...")
    baseline = benchmark(pipeline, "Baseline (bf16)")

    # Try channels_last memory format
    print("\n[3] Applying channels_last memory format...")
    try:
        pipeline.transformer = pipeline.transformer.to(memory_format=torch.channels_last)
        channels_last_time = benchmark(pipeline, "Channels Last")
        print(f"  Speedup: {baseline/channels_last_time:.2f}x")
    except Exception as e:
        print(f"  Failed: {e}")
        channels_last_time = None

    # Reset and try compile with default mode
    print("\n[4] Resetting and trying torch.compile (default)...")

    if not use_compile:
        print("  Skipping torch.compile (not supported on this GPU)")
        compile_time = None
    else:
        del pipeline
        torch.cuda.empty_cache()

        pipeline = QwenImageEditPlusPipeline.from_pretrained(
            "Qwen/Qwen-Image-Edit-2509",
            torch_dtype=dtype,
        ).to('cuda')

        print("  Compiling transformer (this may take a few minutes)...")
        try:
            pipeline.transformer = torch.compile(
                pipeline.transformer,
                mode="default",
                fullgraph=False,  # Allow graph breaks
            )
            compile_time = benchmark(pipeline, "torch.compile (default)", runs=3)
            print(f"  Speedup vs baseline: {baseline/compile_time:.2f}x")
        except Exception as e:
            print(f"  Failed: {e}")
            import traceback
            traceback.print_exc()
            compile_time = None

    # Try compile with max-autotune
    print("\n[5] Trying torch.compile (max-autotune)...")

    if not use_compile:
        print("  Skipping torch.compile (not supported on this GPU)")
        autotune_time = None
    else:
        try:
            del pipeline
        except NameError:
            pass
        torch.cuda.empty_cache()

        pipeline = QwenImageEditPlusPipeline.from_pretrained(
            "Qwen/Qwen-Image-Edit-2509",
            torch_dtype=dtype,
        ).to('cuda')

        print("  Compiling with max-autotune (may take longer)...")
        try:
            pipeline.transformer = torch.compile(
                pipeline.transformer,
                mode="max-autotune",
                fullgraph=False,
            )
            autotune_time = benchmark(pipeline, "torch.compile (max-autotune)", runs=3)
            print(f"  Speedup vs baseline: {baseline/autotune_time:.2f}x")
        except Exception as e:
            print(f"  Failed: {e}")
            import traceback
            traceback.print_exc()
            autotune_time = None

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Baseline (bf16, 4 steps): {baseline:.3f}s")
    if channels_last_time:
        print(f"Channels Last: {channels_last_time:.3f}s ({baseline/channels_last_time:.2f}x)")
    if compile_time:
        print(f"torch.compile (default): {compile_time:.3f}s ({baseline/compile_time:.2f}x)")
    if autotune_time:
        print(f"torch.compile (max-autotune): {autotune_time:.3f}s ({baseline/autotune_time:.2f}x)")


if __name__ == "__main__":
    main()
