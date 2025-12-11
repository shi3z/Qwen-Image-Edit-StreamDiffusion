#!/usr/bin/env python3
"""
FP8 Pipeline for Qwen-Image-Edit-2509

Integrates FP8 quantized transformer into the diffusion pipeline.
Supports both calibrated (pre-computed scales) and dynamic (runtime) quantization.
"""

import os
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')

import torch
import torch.nn as nn
from diffusers import QwenImageEditPlusPipeline
from PIL import Image
import numpy as np
import json
from pathlib import Path
import time
import gc

from fp8_linear import (
    FP8LinearSlow,
    FP8LinearFast,
    replace_linear_with_fp8,
    FP8_E4M3_MAX,
    FP8_E5M2_MAX,
)


def load_fp8_checkpoint(checkpoint_dir: str) -> tuple:
    """
    Load FP8 calibration checkpoint.

    Returns:
        weight_scales: Dict of layer name -> scale info
        activation_scales: Dict of layer name -> activation scale info
    """
    checkpoint_path = Path(checkpoint_dir)

    # Load layer info
    with open(checkpoint_path / "layer_info.json", 'r') as f:
        layer_info = json.load(f)

    # Load weight scales
    weight_scales_pt = torch.load(checkpoint_path / "weight_scales.pt", weights_only=True)

    # Reconstruct weight_scales dict
    weight_scales = {}
    for name, info in layer_info.items():
        scale_key = f"{name}.scale_w"
        if scale_key in weight_scales_pt:
            weight_scales[name] = {
                'scale': weight_scales_pt[scale_key],
                'shape': tuple(info['shape']),
                'absmax': info['absmax']
            }

    # Load activation scales
    with open(checkpoint_path / "activation_scales.json", 'r') as f:
        activation_scales = json.load(f)

    return weight_scales, activation_scales


def compute_weight_scales_dynamic(model: nn.Module, target_modules=(nn.Linear,)) -> dict:
    """
    Compute weight scales dynamically (without calibration).
    Uses per-channel absmax scaling.
    """
    weight_scales = {}

    for name, module in model.named_modules():
        if isinstance(module, target_modules):
            with torch.no_grad():
                w = module.weight.float()
                per_channel_absmax = w.abs().max(dim=1)[0]
                scale = per_channel_absmax / FP8_E4M3_MAX + 1e-12

                weight_scales[name] = {
                    'scale': scale.cpu(),
                    'shape': tuple(w.shape),
                    'absmax': per_channel_absmax.max().item()
                }

    return weight_scales


def create_fp8_pipeline(
    model_name: str = "Qwen/Qwen-Image-Edit-2509",
    checkpoint_dir: str = None,
    use_fast: bool = True,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> QwenImageEditPlusPipeline:
    """
    Create FP8-quantized pipeline for Qwen-Image-Edit.

    Args:
        model_name: HuggingFace model name
        checkpoint_dir: Path to calibration checkpoint (None for dynamic)
        use_fast: Use FP8LinearFast (True) or FP8LinearSlow (False)
        device: Target device
        dtype: Base dtype for non-quantized components

    Returns:
        Pipeline with FP8-quantized transformer
    """
    print(f"Loading base model...")
    pipeline = QwenImageEditPlusPipeline.from_pretrained(
        model_name,
        torch_dtype=dtype
    ).to(device)

    transformer = pipeline.transformer

    # Get scales
    if checkpoint_dir and Path(checkpoint_dir).exists():
        print(f"Loading calibration from {checkpoint_dir}...")
        weight_scales, activation_scales = load_fp8_checkpoint(checkpoint_dir)
    else:
        print("Computing dynamic weight scales...")
        weight_scales = compute_weight_scales_dynamic(transformer)
        activation_scales = {}  # Will use dynamic activation scaling

    # Count original Linear layers
    original_count = sum(1 for m in transformer.modules() if isinstance(m, nn.Linear))
    print(f"Found {original_count} Linear layers in transformer")

    # Replace with FP8
    print(f"Replacing Linear layers with FP8Linear ({'fast' if use_fast else 'slow'})...")
    replaced = replace_linear_with_fp8(
        transformer,
        weight_scales,
        activation_scales,
        use_fast=use_fast,
    )
    print(f"Replaced {replaced} layers")

    # Clear cache
    torch.cuda.empty_cache()
    gc.collect()

    return pipeline


def benchmark_fp8_pipeline(
    pipeline: QwenImageEditPlusPipeline,
    num_warmup: int = 2,
    num_runs: int = 3,
    num_steps: int = 4,
):
    """
    Benchmark FP8 pipeline performance.
    """
    device = next(pipeline.transformer.parameters()).device

    # Create test image
    test_img = Image.fromarray(
        np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    )
    prompt = "Transform this into an oil painting style"

    # Warmup
    print(f"Warming up ({num_warmup} runs)...")
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
            )
        torch.cuda.synchronize()

    # Benchmark
    print(f"Benchmarking ({num_runs} runs, {num_steps} steps)...")
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
            )

        torch.cuda.synchronize()
        elapsed = time.time() - start
        times.append(elapsed)
        print(f"  Run {i+1}: {elapsed:.3f}s")

    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)

    return {
        'avg_time': avg_time,
        'min_time': min_time,
        'max_time': max_time,
        'times': times,
        'result': result,
    }


def compare_outputs(
    fp8_pipeline: QwenImageEditPlusPipeline,
    bf16_pipeline: QwenImageEditPlusPipeline,
    num_samples: int = 3,
):
    """
    Compare FP8 vs BF16 output quality.
    """
    device = next(fp8_pipeline.transformer.parameters()).device

    prompts = [
        "Transform this into an oil painting style",
        "Make it look like a watercolor painting",
        "Convert to anime style",
    ]

    results = []

    for i in range(min(num_samples, len(prompts))):
        test_img = Image.fromarray(
            np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        )
        prompt = prompts[i]

        print(f"Sample {i+1}: {prompt[:40]}...")

        seed = 42 + i

        with torch.no_grad():
            # BF16 output
            bf16_result = bf16_pipeline(
                image=[test_img],
                prompt=prompt,
                generator=torch.Generator(device=device).manual_seed(seed),
                true_cfg_scale=4.0,
                negative_prompt=" ",
                num_inference_steps=4,
                guidance_scale=1.0,
            )
            bf16_img = np.array(bf16_result.images[0]).astype(np.float32)

            # FP8 output
            fp8_result = fp8_pipeline(
                image=[test_img],
                prompt=prompt,
                generator=torch.Generator(device=device).manual_seed(seed),
                true_cfg_scale=4.0,
                negative_prompt=" ",
                num_inference_steps=4,
                guidance_scale=1.0,
            )
            fp8_img = np.array(fp8_result.images[0]).astype(np.float32)

        # Compute metrics
        mse = np.mean((bf16_img - fp8_img) ** 2)
        psnr = 10 * np.log10(255**2 / mse) if mse > 0 else float('inf')
        max_diff = np.abs(bf16_img - fp8_img).max()

        results.append({
            'prompt': prompt,
            'mse': mse,
            'psnr': psnr,
            'max_diff': max_diff,
        })

        print(f"    MSE: {mse:.2f}, PSNR: {psnr:.2f} dB, Max diff: {max_diff:.1f}")

    return results


def get_memory_usage():
    """Get current GPU memory usage in GB"""
    if torch.cuda.is_available():
        return {
            'allocated': torch.cuda.memory_allocated() / 1e9,
            'reserved': torch.cuda.memory_reserved() / 1e9,
            'max_allocated': torch.cuda.max_memory_allocated() / 1e9,
        }
    return {'allocated': 0, 'reserved': 0, 'max_allocated': 0}


def main():
    print("=" * 70)
    print("FP8 Pipeline Benchmark for Qwen-Image-Edit-2509")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"\n[1] Environment")
    print(f"  Device: {device}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        props = torch.cuda.get_device_properties(0)
        print(f"  Compute Capability: {props.major}.{props.minor}")
        print(f"  Total Memory: {props.total_memory / 1e9:.2f} GB")

    # Check FP8 support
    print(f"\n[2] FP8 Support Check")
    try:
        test_fp8 = torch.tensor([1.0], dtype=torch.float8_e4m3fn, device=device)
        print(f"  torch.float8_e4m3fn: Supported")
        test_fp8 = torch.tensor([1.0], dtype=torch.float8_e5m2, device=device)
        print(f"  torch.float8_e5m2: Supported")
    except Exception as e:
        print(f"  FP8 dtype not supported: {e}")
        return

    # Check scaled_mm (use 16-divisible dimensions required by cuBLAS FP8)
    # cuBLASLt requires row-major x column-major layout:
    # - First matrix: row-major (contiguous)
    # - Second matrix: column-major (non-contiguous via .T, NOT .T.contiguous())
    scaled_mm_available = False
    try:
        if hasattr(torch, '_scaled_mm'):
            # Create row-major matrix a (16x32)
            a = torch.randn(16, 32, device=device).to(torch.float8_e5m2)
            # Create column-major matrix b via .T (NOT .T.contiguous()!)
            b_base = torch.randn(16, 32, device=device).to(torch.float8_e4m3fn)
            b = b_base.T  # This creates column-major (non-contiguous)
            scale_a = torch.tensor(1.0, device=device)
            scale_b = torch.tensor(1.0, device=device)
            _ = torch._scaled_mm(a, b, scale_a=scale_a, scale_b=scale_b, out_dtype=torch.bfloat16)
            scaled_mm_available = True
            print(f"  torch._scaled_mm: Supported")
    except Exception as e:
        print(f"  torch._scaled_mm: Not available ({e})")

    # Clear memory
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats()

    # Test FP8LinearSlow first (should always work)
    print(f"\n[3] Loading FP8 Pipeline (slow/validation mode)...")
    torch.cuda.reset_peak_memory_stats()

    fp8_pipeline_slow = create_fp8_pipeline(
        checkpoint_dir="fp8_checkpoint" if Path("fp8_checkpoint").exists() else None,
        use_fast=False,
        device=device,
    )

    mem_fp8_slow = get_memory_usage()
    print(f"  Memory allocated: {mem_fp8_slow['allocated']:.2f} GB")
    print(f"  Memory peak: {mem_fp8_slow['max_allocated']:.2f} GB")

    # Benchmark FP8 slow
    print(f"\n[4] Benchmarking FP8 Pipeline (slow mode)...")
    results_slow = benchmark_fp8_pipeline(fp8_pipeline_slow, num_warmup=2, num_runs=3)
    print(f"  Average: {results_slow['avg_time']:.3f}s")
    print(f"  Min: {results_slow['min_time']:.3f}s")
    print(f"  Max: {results_slow['max_time']:.3f}s")

    # Save sample output
    results_slow['result'].images[0].save("fp8_slow_output.jpg")
    print(f"  Sample saved to fp8_slow_output.jpg")

    # Clean up
    del fp8_pipeline_slow
    torch.cuda.empty_cache()
    gc.collect()

    # Test FP8LinearFast if scaled_mm is available
    if scaled_mm_available:
        print(f"\n[5] Loading FP8 Pipeline (fast mode)...")
        torch.cuda.reset_peak_memory_stats()

        fp8_pipeline_fast = create_fp8_pipeline(
            checkpoint_dir="fp8_checkpoint" if Path("fp8_checkpoint").exists() else None,
            use_fast=True,
            device=device,
        )

        mem_fp8_fast = get_memory_usage()
        print(f"  Memory allocated: {mem_fp8_fast['allocated']:.2f} GB")
        print(f"  Memory peak: {mem_fp8_fast['max_allocated']:.2f} GB")

        # Benchmark FP8 fast
        print(f"\n[6] Benchmarking FP8 Pipeline (fast mode)...")
        results_fast = benchmark_fp8_pipeline(fp8_pipeline_fast, num_warmup=2, num_runs=3)
        print(f"  Average: {results_fast['avg_time']:.3f}s")
        print(f"  Min: {results_fast['min_time']:.3f}s")
        print(f"  Max: {results_fast['max_time']:.3f}s")

        # Save sample output
        results_fast['result'].images[0].save("fp8_fast_output.jpg")
        print(f"  Sample saved to fp8_fast_output.jpg")

        del fp8_pipeline_fast
        torch.cuda.empty_cache()
        gc.collect()
    else:
        results_fast = None
        print(f"\n[5-6] Skipping fast mode (scaled_mm not available)")

    # Load BF16 baseline for comparison
    print(f"\n[7] Loading BF16 Baseline Pipeline...")
    torch.cuda.reset_peak_memory_stats()

    bf16_pipeline = QwenImageEditPlusPipeline.from_pretrained(
        "Qwen/Qwen-Image-Edit-2509",
        torch_dtype=torch.bfloat16
    ).to(device)

    mem_bf16 = get_memory_usage()
    print(f"  Memory allocated: {mem_bf16['allocated']:.2f} GB")
    print(f"  Memory peak: {mem_bf16['max_allocated']:.2f} GB")

    # Benchmark BF16
    print(f"\n[8] Benchmarking BF16 Baseline...")
    results_bf16 = benchmark_fp8_pipeline(bf16_pipeline, num_warmup=2, num_runs=3)
    print(f"  Average: {results_bf16['avg_time']:.3f}s")
    print(f"  Min: {results_bf16['min_time']:.3f}s")
    print(f"  Max: {results_bf16['max_time']:.3f}s")

    # Save sample output
    results_bf16['result'].images[0].save("bf16_output.jpg")
    print(f"  Sample saved to bf16_output.jpg")

    # Summary
    print(f"\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"\nPerformance (4 steps, 512x512):")
    print(f"  {'Config':<25} {'Avg Time':<12} {'Speedup':<10}")
    print(f"  {'-'*47}")
    print(f"  {'BF16 Baseline':<25} {results_bf16['avg_time']:.3f}s{'':<6} {'1.00x':<10}")
    print(f"  {'FP8 Slow':<25} {results_slow['avg_time']:.3f}s{'':<6} {results_bf16['avg_time']/results_slow['avg_time']:.2f}x")
    if results_fast:
        print(f"  {'FP8 Fast':<25} {results_fast['avg_time']:.3f}s{'':<6} {results_bf16['avg_time']/results_fast['avg_time']:.2f}x")

    print(f"\nMemory Usage:")
    print(f"  BF16: {mem_bf16['max_allocated']:.2f} GB peak")
    print(f"  FP8 Slow: {mem_fp8_slow['max_allocated']:.2f} GB peak")
    if scaled_mm_available:
        print(f"  FP8 Fast: {mem_fp8_fast['max_allocated']:.2f} GB peak")

    print("=" * 70)


if __name__ == "__main__":
    main()
