#!/usr/bin/env python3
"""
GPU detection utilities for Qwen-Image-Edit-StreamDiffusion
Handles GB10 (Blackwell) GPU detection and JIT/torch.compile compatibility

GB10 Optimization Notes:
- Compute Capability 12.x (Blackwell architecture)
- torch.compile/JIT may have limited support
- Prefer eager mode execution for stability
- FP16/BF16 fully supported
- Tensor Cores available for mixed precision
"""

import torch
import os


def is_gb10_gpu() -> bool:
    """
    Detect if running on NVIDIA GB10 (Blackwell) GPU.

    GB10 has Compute Capability 12.x which currently has issues with
    torch.compile and JIT compilation.

    Returns:
        True if GB10 is detected, False otherwise
    """
    if not torch.cuda.is_available():
        return False

    try:
        device_name = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)

        # Check by name (GB10, Blackwell, Thor, etc.)
        blackwell_names = ["GB10", "Blackwell", "Thor", "GB100", "GB102", "GB200"]
        for name in blackwell_names:
            if name in device_name:
                return True

        # Check by compute capability (12.x is Blackwell)
        if props.major >= 12:
            return True

        return False
    except Exception:
        return False


def get_optimal_dtype_for_gpu() -> torch.dtype:
    """
    Get optimal dtype for current GPU.

    GB10 supports BF16 natively with good performance.
    Falls back to FP16 for older GPUs without BF16 support.
    """
    if not torch.cuda.is_available():
        return torch.float32

    props = torch.cuda.get_device_properties(0)

    # Blackwell (12.x), Hopper (9.x), Ada (8.9), Ampere (8.x) support BF16
    if props.major >= 8:
        return torch.bfloat16
    # Volta (7.x), Turing (7.5) - use FP16
    elif props.major >= 7:
        return torch.float16
    else:
        return torch.float32


def configure_gb10_environment():
    """
    Configure environment variables for GB10 optimization.
    Should be called before importing torch or other CUDA libraries.
    """
    if is_gb10_gpu():
        # Disable JIT compilation for stability
        os.environ.setdefault('PYTORCH_JIT', '0')
        os.environ.setdefault('TORCH_COMPILE_DISABLE', '1')

        # Enable synchronous execution for debugging if needed
        # os.environ.setdefault('CUDA_LAUNCH_BLOCKING', '1')

        # Disable inductor for GB10 (uses JIT internally)
        os.environ.setdefault('TORCHINDUCTOR_DISABLE', '1')

        return True
    return False


def get_gb10_memory_fraction() -> float:
    """
    Get recommended memory fraction for GB10.
    GB10 has unified memory architecture, so we can use more VRAM.
    """
    if not torch.cuda.is_available():
        return 0.8

    props = torch.cuda.get_device_properties(0)
    total_memory_gb = props.total_memory / 1e9

    # GB10 typically has 128GB+ unified memory
    if total_memory_gb >= 64:
        return 0.9
    elif total_memory_gb >= 32:
        return 0.85
    else:
        return 0.8


def should_use_torch_compile() -> bool:
    """
    Determine if torch.compile should be used based on GPU type.

    Returns:
        True if torch.compile is safe to use, False if should be skipped
    """
    # Check environment variable override
    env_disable = os.environ.get('TORCH_COMPILE_DISABLE', '').lower()
    if env_disable in ('1', 'true', 'yes'):
        return False

    env_jit_disable = os.environ.get('PYTORCH_JIT', '').lower()
    if env_jit_disable == '0':
        return False

    # GB10 has issues with torch.compile
    if is_gb10_gpu():
        return False

    return True


def get_gpu_info() -> dict:
    """
    Get detailed GPU information.

    Returns:
        Dictionary with GPU name, compute capability, memory, and compatibility info
    """
    if not torch.cuda.is_available():
        return {
            "available": False,
            "name": "N/A",
            "compute_capability": "N/A",
            "memory_gb": 0,
            "is_gb10": False,
            "torch_compile_safe": False
        }

    try:
        props = torch.cuda.get_device_properties(0)
        device_name = torch.cuda.get_device_name(0)
        is_gb10 = is_gb10_gpu()

        return {
            "available": True,
            "name": device_name,
            "compute_capability": f"{props.major}.{props.minor}",
            "memory_gb": props.total_memory / 1e9,
            "is_gb10": is_gb10,
            "torch_compile_safe": not is_gb10
        }
    except Exception as e:
        return {
            "available": True,
            "name": "Unknown",
            "compute_capability": "Unknown",
            "memory_gb": 0,
            "is_gb10": False,
            "torch_compile_safe": True,
            "error": str(e)
        }


def print_gpu_info():
    """Print GPU information and compatibility status"""
    info = get_gpu_info()

    print(f"[GPU Info]")
    print(f"  CUDA Available: {info['available']}")
    if info['available']:
        print(f"  GPU Name: {info['name']}")
        print(f"  Compute Capability: {info['compute_capability']}")
        print(f"  Memory: {info['memory_gb']:.2f} GB")
        print(f"  Is GB10 (Blackwell): {info['is_gb10']}")
        print(f"  torch.compile Safe: {info['torch_compile_safe']}")


def safe_torch_compile(model, mode="default", fullgraph=False, **kwargs):
    """
    Safely apply torch.compile only if the GPU supports it.

    Args:
        model: The model to compile
        mode: Compilation mode ("default", "reduce-overhead", "max-autotune")
        fullgraph: Whether to require full graph compilation
        **kwargs: Additional arguments to torch.compile

    Returns:
        The compiled model if safe, otherwise the original model unchanged
    """
    if not should_use_torch_compile():
        gpu_info = get_gpu_info()
        gpu_name = gpu_info.get('name', 'Unknown')
        print(f"  Skipping torch.compile (not supported on {gpu_name})")
        return model

    try:
        print(f"  Applying torch.compile (mode={mode})...")
        compiled = torch.compile(model, mode=mode, fullgraph=fullgraph, **kwargs)
        print("  torch.compile applied successfully")
        return compiled
    except Exception as e:
        print(f"  torch.compile failed: {e}")
        return model


if __name__ == "__main__":
    # Test GPU detection
    print("=" * 50)
    print("GPU Compatibility Test")
    print("=" * 50)
    print_gpu_info()
    print()
    print(f"Should use torch.compile: {should_use_torch_compile()}")
