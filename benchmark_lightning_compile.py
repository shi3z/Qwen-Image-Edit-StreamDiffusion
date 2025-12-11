#!/usr/bin/env python3
"""
Benchmark Lightning LoRA + torch.compile combination
GB10 compatible version - torch.compile auto-skipped on Blackwell GPUs
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

def benchmark(pipeline, name, steps=2, runs=5):
    """Run benchmark and return average time"""
    dummy = Image.new('RGB', (512, 512), color='gray')

    # Warmup
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
    print("Lightning LoRA + torch.compile Benchmark")
    print("=" * 60)

    # Print GPU info and check compatibility
    print("\n[0] Checking GPU compatibility...")
    print_gpu_info()
    use_compile = should_use_torch_compile()
    print(f"  torch.compile enabled: {use_compile}")

    # Load base model
    dtype = get_optimal_dtype_for_gpu()
    print(f"\n[1] Loading Qwen-Image-Edit-2509 ({dtype})...")
    pipeline = QwenImageEditPlusPipeline.from_pretrained(
        "Qwen/Qwen-Image-Edit-2509",
        torch_dtype=dtype,
    ).to('cuda')

    mem = torch.cuda.max_memory_allocated() / 1e9
    print(f"  GPU Memory: {mem:.2f} GB")

    # Load Lightning LoRA
    print("\n[2] Loading Lightning LoRA (4-step)...")
    pipeline.load_lora_weights(
        "lightx2v/Qwen-Image-Lightning",
        weight_name="Qwen-Image-Edit-Lightning-4steps-V1.0.safetensors",
    )
    print("  Loaded!")

    # Baseline with LoRA (2 steps, no compile)
    print("\n[3] Lightning LoRA only (2 steps)...")
    lora_only = benchmark(pipeline, "LoRA only (2 steps)", steps=2, runs=3)

    # Apply torch.compile (if supported)
    if use_compile:
        print("\n[4] Applying torch.compile...")
        pipeline.transformer = torch.compile(
            pipeline.transformer,
            mode="default",
            fullgraph=False,
        )
        print("  Compilation setup done!")

        # Warmup for compile (first run triggers JIT)
        print("\n[5] torch.compile warmup (JIT compilation)...")
        dummy = Image.new('RGB', (512, 512), color='gray')
        with torch.no_grad():
            for i in range(3):
                print(f"  Warmup {i+1}/3...")
                start = time.time()
                _ = pipeline(
                    image=[dummy],
                    prompt="test",
                    generator=torch.Generator(device='cuda').manual_seed(42),
                    true_cfg_scale=4.0,
                    negative_prompt=" ",
                    num_inference_steps=2,
                    guidance_scale=1.0,
                )
                torch.cuda.synchronize()
                print(f"    took {time.time()-start:.2f}s")

        # Benchmark with LoRA + compile (2 steps)
        print("\n[6] Lightning LoRA + torch.compile (2 steps)...")
        lora_compile = benchmark(pipeline, "LoRA + compile (2 steps)", steps=2, runs=5)

        print(f"\n  Speedup vs LoRA only: {lora_only/lora_compile:.2f}x")
    else:
        print("\n[4-6] Skipping torch.compile (not supported on this GPU)")
        lora_compile = None

    # Save test image
    print("\n[7] Generating test image...")
    with torch.no_grad():
        result = pipeline(
            image=[dummy],
            prompt="oil painting of a landscape",
            generator=torch.Generator(device='cuda').manual_seed(42),
            true_cfg_scale=4.0,
            negative_prompt=" ",
            num_inference_steps=2,
            guidance_scale=1.0,
        )
    result.images[0].save("/mnt/raid6/project/stream/test_lightning_compile_output.jpg")
    print("  Test image saved!")

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Lightning LoRA only (2 steps): {lora_only:.3f}s")
    if lora_compile is not None:
        print(f"Lightning LoRA + torch.compile (2 steps): {lora_compile:.3f}s")
        print(f"Speedup: {lora_only/lora_compile:.2f}x")
        print(f"\nCompared to original baseline (~9.5s):")
        print(f"  Final speedup: {9.5/lora_compile:.2f}x")
    else:
        print("torch.compile: Skipped (not supported on this GPU)")
        print(f"\nCompared to original baseline (~9.5s):")
        print(f"  Final speedup (LoRA only): {9.5/lora_only:.2f}x")


if __name__ == "__main__":
    main()
