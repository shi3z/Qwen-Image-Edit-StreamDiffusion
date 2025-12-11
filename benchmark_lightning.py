#!/usr/bin/env python3
"""
Benchmark Qwen-Image-Lightning LoRA for faster inference
GB10 compatible version
"""
import os
# Use GPU 0 for GB10
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')

import torch
import time
from PIL import Image
from diffusers import QwenImageEditPlusPipeline
from gpu_utils import get_optimal_dtype_for_gpu, print_gpu_info

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')

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
    print("Qwen-Image-Lightning LoRA Benchmark (GB10 Compatible)")
    print("=" * 60)

    # Print GPU info
    print("\n[0] GPU Information")
    print_gpu_info()

    # Load base model
    dtype = get_optimal_dtype_for_gpu()
    print(f"\n[1] Loading Qwen-Image-Edit-2509 ({dtype})...")
    pipeline = QwenImageEditPlusPipeline.from_pretrained(
        "Qwen/Qwen-Image-Edit-2509",
        torch_dtype=dtype,
    ).to('cuda')

    mem = torch.cuda.max_memory_allocated() / 1e9
    print(f"  GPU Memory: {mem:.2f} GB")

    # Baseline (4 steps)
    print("\n[2] Baseline benchmark (4 steps)...")
    baseline_4 = benchmark(pipeline, "Baseline (4 steps)", steps=4, runs=3)

    # Load Lightning LoRA
    print("\n[3] Loading Lightning LoRA...")
    try:
        # Try 4-step version first
        pipeline.load_lora_weights(
            "lightx2v/Qwen-Image-Lightning",
            weight_name="Qwen-Image-Edit-Lightning-4steps-V1.0.safetensors",
        )
        print("  Loaded 4-step Lightning LoRA!")

        # Benchmark with LoRA (4 steps)
        print("\n[4] Lightning LoRA benchmark (4 steps)...")
        lightning_4 = benchmark(pipeline, "Lightning (4 steps)", steps=4, runs=3)
        print(f"  Speedup vs baseline: {baseline_4/lightning_4:.2f}x")

        # Try 2 steps
        print("\n[5] Lightning LoRA benchmark (2 steps)...")
        lightning_2 = benchmark(pipeline, "Lightning (2 steps)", steps=2, runs=3)
        print(f"  Speedup vs baseline 4-step: {baseline_4/lightning_2:.2f}x")

    except Exception as e:
        print(f"  4-step LoRA failed: {e}")
        print("  Trying 8-step version...")

        try:
            pipeline.load_lora_weights(
                "lightx2v/Qwen-Image-Lightning",
                weight_name="Qwen-Image-Lightning-8steps-V1.0.safetensors",
            )
            print("  Loaded 8-step Lightning LoRA!")

            # Benchmark with LoRA (8 steps)
            print("\n[4] Lightning LoRA benchmark (8 steps)...")
            lightning_8 = benchmark(pipeline, "Lightning (8 steps)", steps=8, runs=3)

            # Also test 4 steps with 8-step LoRA
            print("\n[5] Lightning LoRA benchmark (4 steps with 8-step LoRA)...")
            lightning_4 = benchmark(pipeline, "Lightning 8-step LoRA (4 steps)", steps=4, runs=3)
            print(f"  Speedup vs baseline: {baseline_4/lightning_4:.2f}x")

        except Exception as e2:
            print(f"  8-step LoRA also failed: {e2}")
            import traceback
            traceback.print_exc()

    # Save test image
    print("\n[6] Generating test image...")
    dummy = Image.new('RGB', (512, 512), color='gray')
    with torch.no_grad():
        result = pipeline(
            image=[dummy],
            prompt="oil painting of a landscape",
            generator=torch.Generator(device='cuda').manual_seed(42),
            true_cfg_scale=4.0,
            negative_prompt=" ",
            num_inference_steps=4,
            guidance_scale=1.0,
        )
    result.images[0].save("/mnt/raid6/project/stream/test_lightning_output.jpg")
    print("  Test image saved to test_lightning_output.jpg")

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Baseline (4 steps): {baseline_4:.3f}s")


if __name__ == "__main__":
    main()
