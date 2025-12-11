#!/usr/bin/env python3
"""
Benchmark script for Qwen-Image-Edit-2509 with:
- Full backend FP4 quantization (transformer + text_encoder via torchao)
- 4-step Lightning LoRA for faster inference
"""

import torch
from diffusers import QwenImageEditPlusPipeline
from torchao.quantization import quantize_, FPXWeightOnlyConfig
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

def quantize_model_fp4(model, name="model"):
    """Apply FP4 quantization to a model"""
    try:
        fp4_config = FPXWeightOnlyConfig(ebits=2, mbits=1)
        quantize_(model, fp4_config)
        print(f"    {name}: FP4 quantized successfully")
        return True
    except Exception as e:
        print(f"    {name}: FP4 quantization failed - {e}")
        return False

def main():
    print("=" * 70)
    print("Qwen-Image-Edit-2509 FP4 + Lightning LoRA Benchmark")
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

    print(f"\n[1] Loading model (BF16)...")
    load_start = time.time()

    pipeline = QwenImageEditPlusPipeline.from_pretrained(
        "Qwen/Qwen-Image-Edit-2509",
        torch_dtype=torch.bfloat16
    ).to(device)

    load_time = time.time() - load_start
    alloc_bf16, _, _ = get_gpu_memory_info()
    print(f"    Model load time: {load_time:.2f}s")
    print(f"    GPU Memory (BF16): {alloc_bf16:.2f} GB allocated")

    # Load Lightning LoRA before quantization
    print(f"\n[2] Loading 4-step Lightning LoRA...")
    try:
        pipeline.load_lora_weights(
            "lightx2v/Qwen-Image-Lightning",
            weight_name="Qwen-Image-Edit-Lightning-4steps-V1.0.safetensors",
        )
        print("    4-step Lightning LoRA loaded!")
        lora_loaded = True
    except Exception as e:
        print(f"    Failed to load LoRA: {e}")
        print("    Continuing without LoRA...")
        lora_loaded = False

    # Apply FP4 quantization to all backend components
    print(f"\n[3] Applying FP4 (e2m1) quantization to backend...")
    quant_start = time.time()

    # Quantize transformer (main compute component)
    print("  Quantizing components:")
    quantize_model_fp4(pipeline.transformer, "transformer")

    # Quantize text encoder if available
    if hasattr(pipeline, 'text_encoder') and pipeline.text_encoder is not None:
        quantize_model_fp4(pipeline.text_encoder, "text_encoder")

    # Quantize VAE encoder/decoder if available
    if hasattr(pipeline, 'vae') and pipeline.vae is not None:
        try:
            quantize_model_fp4(pipeline.vae, "vae")
        except Exception as e:
            print(f"    vae: skipped (may not support FP4) - {e}")

    quant_time = time.time() - quant_start
    torch.cuda.empty_cache()
    gc.collect()

    alloc_fp4, _, _ = get_gpu_memory_info()
    print(f"\n    Quantization time: {quant_time:.2f}s")
    print(f"    GPU Memory (FP4): {alloc_fp4:.2f} GB allocated")
    print(f"    Memory reduction: {(alloc_bf16 - alloc_fp4):.2f} GB ({(1 - alloc_fp4/alloc_bf16)*100:.1f}%)")

    # Create test image
    test_img = create_test_image((512, 512))
    prompt = "Transform this into an oil painting style"

    print(f"\n[4] Warmup (2 runs @ 4 steps)...")
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

    # Benchmark different step counts (optimized for Lightning LoRA)
    print(f"\n[5] Benchmarking FP4 + Lightning LoRA (3 runs each)...")
    print("-" * 70)
    print(f"{'Steps':<10} {'Avg Time':<12} {'Min Time':<12} {'Max Time':<12} {'FPS':<10} {'ms/step':<10}")
    print("-" * 70)

    results = []
    step_counts = [2, 4] if lora_loaded else [2, 4, 6, 8]

    for steps in step_counts:
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
    print(f"\n[6] Final GPU Memory:")
    print(f"    Allocated: {alloc:.2f} GB")
    print(f"    Max Allocated: {max_alloc:.2f} GB")

    # Summary
    print(f"\n[Summary]")
    print("=" * 70)
    best = min(results, key=lambda x: x['avg_time'])
    print(f"  Configuration: FP4 (e2m1) + Lightning LoRA {'(4-step)' if lora_loaded else '(not loaded)'}")
    print(f"  Fastest: {best['steps']} steps @ {best['fps']:.3f} FPS ({best['avg_time']:.3f}s/image)")
    print(f"  Memory: {alloc:.2f} GB (reduced from {alloc_bf16:.2f} GB BF16)")
    print("=" * 70)

if __name__ == "__main__":
    main()
