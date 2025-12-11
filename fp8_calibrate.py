#!/usr/bin/env python3
"""
FP8 Calibration Script for Qwen-Image-Edit-2509
Collects activation statistics and computes per-channel scales for FP8 quantization.

FP8 Format Specification:
- Weights: FP8 e4m3 (4 exponent bits, 3 mantissa bits) - better resolution
- Activations: FP8 e5m2 (5 exponent bits, 2 mantissa bits) - better range
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
from collections import defaultdict
import gc

# FP8 constants
FP8_E4M3_MAX = 448.0  # max representable value in e4m3
FP8_E5M2_MAX = 57344.0  # max representable value in e5m2


class ActivationCollector:
    """Collects activation statistics during forward pass"""

    def __init__(self):
        self.stats = defaultdict(lambda: {
            'max': [],
            'min': [],
            'absmax': [],
            'mean': [],
            'std': [],
            'count': 0
        })
        self.hooks = []

    def _hook_fn(self, name):
        def hook(module, input, output):
            if isinstance(input, tuple):
                x = input[0]
            else:
                x = input

            if x is None or not isinstance(x, torch.Tensor):
                return

            with torch.no_grad():
                x_flat = x.float().flatten()
                self.stats[name]['max'].append(x_flat.max().item())
                self.stats[name]['min'].append(x_flat.min().item())
                self.stats[name]['absmax'].append(x_flat.abs().max().item())
                self.stats[name]['mean'].append(x_flat.mean().item())
                self.stats[name]['std'].append(x_flat.std().item())
                self.stats[name]['count'] += 1
        return hook

    def register_hooks(self, model, target_modules=(nn.Linear,)):
        """Register forward hooks on target modules"""
        for name, module in model.named_modules():
            if isinstance(module, target_modules):
                hook = module.register_forward_hook(self._hook_fn(name))
                self.hooks.append(hook)
        print(f"Registered {len(self.hooks)} hooks")

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def get_aggregated_stats(self):
        """Aggregate collected statistics"""
        aggregated = {}
        for name, stat in self.stats.items():
            if stat['count'] > 0:
                aggregated[name] = {
                    'absmax': max(stat['absmax']),
                    'max': max(stat['max']),
                    'min': min(stat['min']),
                    'mean': np.mean(stat['mean']),
                    'std': np.mean(stat['std']),
                    'samples': stat['count']
                }
        return aggregated


def compute_weight_scales(model, target_modules=(nn.Linear,)):
    """
    Compute per-channel (per-output-feature) scales for weights.
    For Linear layer weight shape (out_features, in_features),
    scale is computed per out_features dimension.
    """
    weight_scales = {}

    for name, module in model.named_modules():
        if isinstance(module, target_modules):
            with torch.no_grad():
                w = module.weight.float()
                # Per-channel (per-row) absmax for Linear weights
                # Shape: (out_features,)
                per_channel_absmax = w.abs().max(dim=1)[0]
                # Add small epsilon to avoid division by zero
                scale = per_channel_absmax / FP8_E4M3_MAX + 1e-12

                weight_scales[name] = {
                    'scale': scale.cpu(),
                    'shape': tuple(w.shape),
                    'absmax': per_channel_absmax.max().item()
                }

    return weight_scales


def compute_activation_scales(activation_stats):
    """
    Compute per-tensor scales for activations based on collected statistics.
    """
    activation_scales = {}

    for name, stats in activation_stats.items():
        absmax = stats['absmax']
        # Scale to fit in e5m2 range with some headroom
        scale = absmax / FP8_E5M2_MAX + 1e-12

        activation_scales[name] = {
            'scale': scale,
            'absmax': absmax,
            'format': 'e5m2'
        }

    return activation_scales


def quantize_weight_to_fp8(weight, scale, fmt='e4m3'):
    """
    Quantize weight tensor to FP8 representation.
    Returns uint8 tensor (storage) and scale.

    Args:
        weight: (out_features, in_features) float tensor
        scale: (out_features,) scale tensor
        fmt: 'e4m3' or 'e5m2'
    """
    if fmt == 'e4m3':
        max_val = FP8_E4M3_MAX
        dtype = torch.float8_e4m3fn
    else:
        max_val = FP8_E5M2_MAX
        dtype = torch.float8_e5m2

    with torch.no_grad():
        # Expand scale for broadcasting: (out_features,) -> (out_features, 1)
        scale_expanded = scale.unsqueeze(1).to(weight.device)

        # Quantize: w_scaled = w / scale
        w_scaled = weight.float() / scale_expanded

        # Clamp to FP8 range
        w_clamped = torch.clamp(w_scaled, -max_val, max_val)

        # Convert to FP8
        w_fp8 = w_clamped.to(dtype)

        return w_fp8, scale


def create_calibration_samples(num_samples=16, size=(512, 512)):
    """Create calibration samples"""
    images = []
    prompts = [
        "Transform this into an oil painting style",
        "Make it look like a watercolor painting",
        "Add a sunset glow to the image",
        "Convert to anime style",
        "Make it look like a sketch",
        "Add snow to the scene",
        "Make it look vintage",
        "Add dramatic lighting",
        "Make it look like pixel art",
        "Add fog effect",
        "Make it look cinematic",
        "Convert to black and white",
        "Add neon colors",
        "Make it look dreamy",
        "Add autumn colors",
        "Make it look futuristic",
    ]

    for i in range(num_samples):
        # Create varied test images
        img = np.random.randint(0, 255, (*size, 3), dtype=np.uint8)
        images.append(Image.fromarray(img))

    return images[:num_samples], prompts[:num_samples]


def run_calibration(pipeline, images, prompts, collector, num_steps=4):
    """Run calibration passes to collect activation statistics"""
    device = next(pipeline.transformer.parameters()).device

    print(f"Running calibration with {len(images)} samples...")

    for i, (img, prompt) in enumerate(zip(images, prompts)):
        print(f"  Sample {i+1}/{len(images)}: {prompt[:40]}...")

        with torch.no_grad():
            _ = pipeline(
                image=[img],
                prompt=prompt,
                generator=torch.Generator(device=device).manual_seed(42),
                true_cfg_scale=4.0,
                negative_prompt=" ",
                num_inference_steps=num_steps,
                guidance_scale=1.0,
            )

        # Clear cache periodically
        if (i + 1) % 4 == 0:
            torch.cuda.empty_cache()

    print("Calibration complete!")


def save_fp8_checkpoint(
    model,
    weight_scales,
    activation_scales,
    output_dir="fp8_checkpoint"
):
    """
    Save FP8-quantized weights and scales.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Quantize and save weights
    fp8_weights = {}
    scales_dict = {}

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and name in weight_scales:
            scale = weight_scales[name]['scale']
            w_fp8, _ = quantize_weight_to_fp8(
                module.weight,
                scale.to(module.weight.device),
                fmt='e4m3'
            )

            fp8_weights[f"{name}.weight_fp8"] = w_fp8.cpu()
            scales_dict[f"{name}.scale_w"] = scale.cpu()

            if module.bias is not None:
                # Keep bias in original precision
                fp8_weights[f"{name}.bias"] = module.bias.cpu()

    # Save FP8 weights
    torch.save(fp8_weights, output_path / "fp8_weights.pt")
    print(f"Saved FP8 weights to {output_path / 'fp8_weights.pt'}")

    # Save weight scales
    torch.save(scales_dict, output_path / "weight_scales.pt")
    print(f"Saved weight scales to {output_path / 'weight_scales.pt'}")

    # Save activation scales as JSON (for inspection)
    activation_scales_serializable = {}
    for name, data in activation_scales.items():
        activation_scales_serializable[name] = {
            'scale': float(data['scale']),
            'absmax': float(data['absmax']),
            'format': data['format']
        }

    with open(output_path / "activation_scales.json", 'w') as f:
        json.dump(activation_scales_serializable, f, indent=2)
    print(f"Saved activation scales to {output_path / 'activation_scales.json'}")

    # Save layer info
    layer_info = {}
    for name, data in weight_scales.items():
        layer_info[name] = {
            'shape': list(data['shape']),
            'absmax': float(data['absmax']),
            'format_w': 'e4m3'
        }

    with open(output_path / "layer_info.json", 'w') as f:
        json.dump(layer_info, f, indent=2)
    print(f"Saved layer info to {output_path / 'layer_info.json'}")

    return output_path


def main():
    print("=" * 70)
    print("FP8 Calibration for Qwen-Image-Edit-2509")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"\n[1] Environment")
    print(f"  Device: {device}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        props = torch.cuda.get_device_properties(0)
        print(f"  Compute Capability: {props.major}.{props.minor}")

    # Check FP8 support
    print(f"\n[2] FP8 Support Check")
    try:
        test_fp8 = torch.tensor([1.0], dtype=torch.float8_e4m3fn, device=device)
        print(f"  torch.float8_e4m3fn: Supported")
        test_fp8 = torch.tensor([1.0], dtype=torch.float8_e5m2, device=device)
        print(f"  torch.float8_e5m2: Supported")
    except Exception as e:
        print(f"  FP8 not supported: {e}")
        return

    # Load model
    print(f"\n[3] Loading model (BF16)...")
    pipeline = QwenImageEditPlusPipeline.from_pretrained(
        "Qwen/Qwen-Image-Edit-2509",
        torch_dtype=torch.bfloat16
    ).to(device)

    transformer = pipeline.transformer

    # Count Linear layers
    linear_count = sum(1 for m in transformer.modules() if isinstance(m, nn.Linear))
    print(f"  Found {linear_count} Linear layers in transformer")

    # Create activation collector
    print(f"\n[4] Setting up activation collection...")
    collector = ActivationCollector()
    collector.register_hooks(transformer)

    # Create calibration samples
    print(f"\n[5] Creating calibration samples...")
    images, prompts = create_calibration_samples(num_samples=8)

    # Run calibration
    print(f"\n[6] Running calibration forward passes...")
    run_calibration(pipeline, images, prompts, collector, num_steps=2)

    # Remove hooks
    collector.remove_hooks()

    # Aggregate activation statistics
    print(f"\n[7] Computing scales...")
    activation_stats = collector.get_aggregated_stats()
    print(f"  Collected stats for {len(activation_stats)} layers")

    # Compute weight scales
    weight_scales = compute_weight_scales(transformer)
    print(f"  Computed weight scales for {len(weight_scales)} layers")

    # Compute activation scales
    activation_scales = compute_activation_scales(activation_stats)
    print(f"  Computed activation scales for {len(activation_scales)} layers")

    # Save checkpoint
    print(f"\n[8] Saving FP8 checkpoint...")
    output_path = save_fp8_checkpoint(
        transformer,
        weight_scales,
        activation_scales,
        output_dir="fp8_checkpoint"
    )

    # Summary statistics
    print(f"\n[9] Summary")
    print("=" * 70)

    # Weight scale statistics
    w_scales = [data['absmax'] for data in weight_scales.values()]
    print(f"  Weight absmax: min={min(w_scales):.4f}, max={max(w_scales):.4f}, mean={np.mean(w_scales):.4f}")

    # Activation scale statistics
    a_scales = [data['absmax'] for data in activation_scales.values()]
    print(f"  Activation absmax: min={min(a_scales):.4f}, max={max(a_scales):.4f}, mean={np.mean(a_scales):.4f}")

    # Estimate memory savings
    original_size = sum(p.numel() * 2 for p in transformer.parameters())  # BF16 = 2 bytes
    fp8_size = sum(p.numel() for p in transformer.parameters())  # FP8 = 1 byte
    print(f"  Original size (BF16): {original_size / 1e9:.2f} GB")
    print(f"  FP8 size estimate: {fp8_size / 1e9:.2f} GB")
    print(f"  Memory reduction: {(1 - fp8_size/original_size) * 100:.1f}%")

    print(f"\n  Checkpoint saved to: {output_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
