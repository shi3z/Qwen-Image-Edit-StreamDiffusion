#!/usr/bin/env python3
"""
Simple benchmark: 2 steps, no CFG (Lightning LoRA)
Optimized for GB10
"""
import os
os.environ.setdefault('PYTORCH_JIT', '0')
os.environ.setdefault('TORCH_COMPILE_DISABLE', '1')
os.environ.setdefault('TORCHINDUCTOR_DISABLE', '1')

import torch
import time
from PIL import Image
from diffusers import QwenImageEditPlusPipeline
from gpu_utils import get_optimal_dtype_for_gpu, print_gpu_info

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')

def main():
    print("=" * 60)
    print("GB10 Benchmark: 2 steps, no CFG (Lightning LoRA)")
    print("=" * 60)

    print_gpu_info()

    dtype = get_optimal_dtype_for_gpu()
    print(f"\nLoading model ({dtype})...")

    pipeline = QwenImageEditPlusPipeline.from_pretrained(
        "Qwen/Qwen-Image-Edit-2509",
        torch_dtype=dtype,
    ).to('cuda')

    print(f"GPU Memory after load: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

    # Load Lightning LoRA
    print("Loading Lightning LoRA...")
    pipeline.load_lora_weights(
        "lightx2v/Qwen-Image-Lightning",
        weight_name="Qwen-Image-Edit-Lightning-4steps-V1.0.safetensors",
    )
    print("Lightning LoRA loaded!")

    # Test image
    dummy = Image.new('RGB', (512, 512), color='gray')

    # Warmup
    print("\nWarmup (2 runs)...")
    for i in range(2):
        with torch.no_grad():
            _ = pipeline(
                image=[dummy],
                prompt="test",
                generator=torch.Generator(device='cuda').manual_seed(42),
                num_inference_steps=2,
                guidance_scale=3.5,
            )
        torch.cuda.synchronize()
        print(f"  Warmup {i+1} done")

    # Benchmark
    print("\nBenchmark: 2 steps, no CFG (5 runs)...")
    print("-" * 40)

    times = []
    for i in range(5):
        torch.cuda.synchronize()
        start = time.time()

        with torch.no_grad():
            _ = pipeline(
                image=[dummy],
                prompt="oil painting style",
                generator=torch.Generator(device='cuda').manual_seed(42),
                num_inference_steps=2,
                guidance_scale=3.5,
            )

        torch.cuda.synchronize()
        elapsed = time.time() - start
        times.append(elapsed)
        print(f"  Run {i+1}: {elapsed:.3f}s")

    print("-" * 40)
    avg = sum(times) / len(times)
    min_t = min(times)
    max_t = max(times)
    fps = 1.0 / avg

    print(f"\nResults:")
    print(f"  Average: {avg:.3f}s")
    print(f"  Min: {min_t:.3f}s")
    print(f"  Max: {max_t:.3f}s")
    print(f"  FPS: {fps:.2f}")
    print(f"  Peak memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

if __name__ == "__main__":
    main()
