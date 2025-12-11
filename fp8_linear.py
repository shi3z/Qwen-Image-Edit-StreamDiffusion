#!/usr/bin/env python3
"""
FP8 Linear Module Implementation for Qwen-Image-Edit-2509

Two implementations:
1. FP8LinearSlow: Reference implementation using dequantize->BF16 GEMM (for validation)
2. FP8LinearFast: Uses torch._scaled_mm for native FP8 GEMM on supported hardware

FP8 Format:
- Weights: e4m3fn (4 exponent bits, 3 mantissa bits) - better resolution
- Activations: e5m2 (5 exponent bits, 2 mantissa bits) - better range
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

# FP8 constants
FP8_E4M3_MAX = 448.0
FP8_E5M2_MAX = 57344.0


def quantize_to_fp8_e4m3(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """
    Quantize tensor to FP8 e4m3 format.

    Args:
        x: Input tensor (any shape)
        scale: Scale tensor (broadcastable to x)

    Returns:
        FP8 e4m3 tensor
    """
    x_scaled = x.float() / scale.float()
    x_clamped = torch.clamp(x_scaled, -FP8_E4M3_MAX, FP8_E4M3_MAX)
    return x_clamped.to(torch.float8_e4m3fn)


def quantize_to_fp8_e5m2(x: torch.Tensor, scale: float) -> torch.Tensor:
    """
    Quantize tensor to FP8 e5m2 format.

    Args:
        x: Input tensor (any shape)
        scale: Scalar scale value

    Returns:
        FP8 e5m2 tensor
    """
    x_scaled = x.float() / scale
    x_clamped = torch.clamp(x_scaled, -FP8_E5M2_MAX, FP8_E5M2_MAX)
    return x_clamped.to(torch.float8_e5m2)


def dequantize_fp8_e4m3(x_fp8: torch.Tensor, scale: torch.Tensor, target_dtype=torch.bfloat16) -> torch.Tensor:
    """
    Dequantize FP8 e4m3 tensor back to higher precision.

    Args:
        x_fp8: FP8 e4m3 tensor
        scale: Scale tensor
        target_dtype: Output dtype (default: bfloat16)

    Returns:
        Dequantized tensor
    """
    return (x_fp8.float() * scale.float()).to(target_dtype)


def dequantize_fp8_e5m2(x_fp8: torch.Tensor, scale: float, target_dtype=torch.bfloat16) -> torch.Tensor:
    """
    Dequantize FP8 e5m2 tensor back to higher precision.

    Args:
        x_fp8: FP8 e5m2 tensor
        scale: Scalar scale value
        target_dtype: Output dtype (default: bfloat16)

    Returns:
        Dequantized tensor
    """
    return (x_fp8.float() * scale).to(target_dtype)


class FP8LinearSlow(nn.Module):
    """
    Reference FP8 Linear implementation.
    Dequantizes weights to BF16 before GEMM.
    Slow but correct - use for validation.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # FP8 quantized weight storage
        self.register_buffer(
            'weight_fp8',
            torch.zeros(out_features, in_features, dtype=torch.float8_e4m3fn, device=device)
        )

        # Per-channel (per-output) weight scale
        self.register_buffer(
            'scale_w',
            torch.ones(out_features, dtype=torch.float32, device=device)
        )

        # Per-tensor activation scale (updated during forward or calibration)
        self.register_buffer(
            'scale_x',
            torch.ones(1, dtype=torch.float32, device=device)
        )

        # Bias stays in BF16
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.bfloat16, device=device))
        else:
            self.register_parameter('bias', None)

        # Dynamic activation scaling flag
        self.dynamic_scale = True

    @classmethod
    def from_linear(cls, linear: nn.Linear, weight_scale: torch.Tensor, activation_scale: float = 1.0):
        """
        Create FP8LinearSlow from an existing nn.Linear module.

        Args:
            linear: Source nn.Linear module
            weight_scale: Per-channel scale for weights (out_features,)
            activation_scale: Per-tensor scale for activations

        Returns:
            FP8LinearSlow module
        """
        has_bias = linear.bias is not None
        fp8_linear = cls(
            linear.in_features,
            linear.out_features,
            bias=has_bias,
            device=linear.weight.device
        )

        # Quantize weights
        with torch.no_grad():
            scale_expanded = weight_scale.unsqueeze(1).to(linear.weight.device)
            w_scaled = linear.weight.float() / scale_expanded
            w_clamped = torch.clamp(w_scaled, -FP8_E4M3_MAX, FP8_E4M3_MAX)
            fp8_linear.weight_fp8.copy_(w_clamped.to(torch.float8_e4m3fn))
            fp8_linear.scale_w.copy_(weight_scale.to(linear.weight.device))
            fp8_linear.scale_x.fill_(activation_scale)

            if has_bias:
                fp8_linear.bias.copy_(linear.bias.to(torch.bfloat16))

        return fp8_linear

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with FP8 weights (dequantized path).

        Args:
            x: Input tensor (..., in_features), typically BF16

        Returns:
            Output tensor (..., out_features) in BF16
        """
        # Dequantize weights: (out_features, in_features)
        # scale_w shape: (out_features,) -> (out_features, 1)
        scale_expanded = self.scale_w.unsqueeze(1)
        w_bf16 = (self.weight_fp8.float() * scale_expanded).to(torch.bfloat16)

        # Standard linear operation
        return F.linear(x.to(torch.bfloat16), w_bf16, self.bias)


class FP8LinearFast(nn.Module):
    """
    Fast FP8 Linear implementation using torch._scaled_mm.
    Requires hardware support for FP8 tensor cores.

    Note: cuBLASLt requires row-major x column-major layout for FP8 matmul.
    We store weight as (out, in) and use .T at runtime to get column-major view.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # FP8 quantized weight storage - stored as (out, in)
        # We use .T at runtime to get column-major (in, out) view for scaled_mm
        self.register_buffer(
            'weight_fp8',
            torch.zeros(out_features, in_features, dtype=torch.float8_e4m3fn, device=device)
        )

        # Per-channel weight scale
        self.register_buffer(
            'scale_w',
            torch.ones(out_features, dtype=torch.float32, device=device)
        )

        # Per-tensor activation scale
        self.register_buffer(
            'scale_x',
            torch.ones(1, dtype=torch.float32, device=device)
        )

        # Combined scale for output rescaling
        self.register_buffer(
            'scale_out',
            torch.ones(1, dtype=torch.float32, device=device)
        )

        # Bias stays in BF16
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.bfloat16, device=device))
        else:
            self.register_parameter('bias', None)

        self.dynamic_scale = True
        self._scaled_mm_available = self._check_scaled_mm()

    def _check_scaled_mm(self) -> bool:
        """Check if torch._scaled_mm is available"""
        try:
            # Check if function exists
            if not hasattr(torch, '_scaled_mm'):
                return False

            # Try a test with 16-divisible dimensions (required by cuBLAS FP8 kernels)
            # cuBLASLt requires row-major x column-major layout:
            # - First matrix: row-major (contiguous)
            # - Second matrix: column-major (non-contiguous via .T, NOT .T.contiguous())
            device = self.weight_fp8.device if self.weight_fp8.device.type == 'cuda' else 'cuda'
            # Create row-major matrix a (16x32)
            a = torch.randn(16, 32, device=device).to(torch.float8_e5m2)
            # Create column-major matrix b by .T (NOT .T.contiguous()!)
            # b_base is (16, 32), .T makes it (32, 16) with stride (1, 32) = column-major
            b_base = torch.randn(16, 32, device=device).to(torch.float8_e4m3fn)
            b = b_base.T  # This is column-major (non-contiguous)
            scale_a = torch.tensor(1.0, device=device)
            scale_b = torch.tensor(1.0, device=device)
            _ = torch._scaled_mm(a, b, scale_a=scale_a, scale_b=scale_b, out_dtype=torch.bfloat16)
            return True
        except Exception as e:
            print(f"torch._scaled_mm not available: {e}")
            return False

    @classmethod
    def from_linear(cls, linear: nn.Linear, weight_scale: torch.Tensor, activation_scale: float = 1.0):
        """
        Create FP8LinearFast from an existing nn.Linear module.

        Args:
            linear: Source nn.Linear module
            weight_scale: Per-channel scale for weights (out_features,)
            activation_scale: Per-tensor scale for activations

        Returns:
            FP8LinearFast module
        """
        has_bias = linear.bias is not None
        fp8_linear = cls(
            linear.in_features,
            linear.out_features,
            bias=has_bias,
            device=linear.weight.device
        )

        # Quantize weights - store as (out, in), use .T at runtime for column-major
        with torch.no_grad():
            # Original weight: (out_features, in_features)
            w = linear.weight.float()  # (out, in)

            # Per-channel scale: (out_features,) -> (out, 1) for broadcasting
            scale_expanded = weight_scale.unsqueeze(1).to(linear.weight.device)  # (out, 1)

            # Scale the weight (keep original shape)
            w_scaled = w / scale_expanded  # Broadcasting: (out, in) / (out, 1)
            w_clamped = torch.clamp(w_scaled, -FP8_E4M3_MAX, FP8_E4M3_MAX)

            fp8_linear.weight_fp8.copy_(w_clamped.to(torch.float8_e4m3fn))
            fp8_linear.scale_w.copy_(weight_scale.to(linear.weight.device))
            fp8_linear.scale_x.fill_(activation_scale)

            # Pre-compute output scale
            fp8_linear.scale_out.fill_(activation_scale)  # Will be multiplied by scale_w in forward

            if has_bias:
                fp8_linear.bias.copy_(linear.bias.to(torch.bfloat16))

        return fp8_linear

    def _compute_dynamic_scale(self, x: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """Compute dynamic activation scale"""
        with torch.no_grad():
            absmax = x.abs().max().item()
            scale = absmax / FP8_E5M2_MAX + 1e-12
        return scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass using native FP8 GEMM.

        Args:
            x: Input tensor (batch, seq_len, in_features) or (batch, in_features)

        Returns:
            Output tensor in BF16
        """
        original_shape = x.shape
        # Flatten to 2D: (batch * seq, in_features)
        if x.dim() > 2:
            x_2d = x.view(-1, self.in_features)
        else:
            x_2d = x

        if self._scaled_mm_available:
            # Dynamic activation scaling
            if self.dynamic_scale:
                scale_x = self._compute_dynamic_scale(x_2d)
            else:
                scale_x = self.scale_x.item()

            # Quantize activation to FP8 e5m2
            x_scaled = x_2d.float() / scale_x
            x_clamped = torch.clamp(x_scaled, -FP8_E5M2_MAX, FP8_E5M2_MAX)
            x_fp8 = x_clamped.to(torch.float8_e5m2)

            # For per-channel weight scales, we need to handle differently
            # scaled_mm expects scalar scales, so we post-multiply
            # y = (x_fp8 @ w_fp8) * (scale_x * scale_w)

            # Use scalar scale = 1.0 for now, then apply per-channel scale after
            try:
                scale_a = torch.tensor(1.0, device=x.device, dtype=torch.float32)
                scale_b = torch.tensor(1.0, device=x.device, dtype=torch.float32)

                # Get column-major weight view: (out, in) -> (in, out) with .T
                # This creates a non-contiguous column-major view required by cuBLASLt
                weight_col_major = self.weight_fp8.T  # (in, out) column-major

                # FP8 matmul: (batch, in) @ (in, out) -> (batch, out)
                y_fp32 = torch._scaled_mm(
                    x_fp8,
                    weight_col_major,
                    scale_a=scale_a,
                    scale_b=scale_b,
                    out_dtype=torch.float32
                )

                # Apply scales: (batch, out) * (out,) * scalar
                y = (y_fp32 * self.scale_w.unsqueeze(0) * scale_x).to(torch.bfloat16)

            except Exception as e:
                # Fallback to slow path
                print(f"scaled_mm failed, falling back: {e}")
                return self._forward_slow(x)
        else:
            return self._forward_slow(x)

        # Add bias if present
        if self.bias is not None:
            y = y + self.bias

        # Reshape back
        if len(original_shape) > 2:
            y = y.view(*original_shape[:-1], self.out_features)

        return y

    def _forward_slow(self, x: torch.Tensor) -> torch.Tensor:
        """Fallback slow path using dequantization"""
        # Dequantize weights
        # weight_fp8: (out, in), scale_w: (out,)
        scale_expanded = self.scale_w.unsqueeze(1)  # (out, 1)
        w_bf16 = (self.weight_fp8.float() * scale_expanded).to(torch.bfloat16)  # (out, in)

        return F.linear(x.to(torch.bfloat16), w_bf16, self.bias)


def replace_linear_with_fp8(
    model: nn.Module,
    weight_scales: dict,
    activation_scales: dict,
    use_fast: bool = True,
    exclude_patterns: list = None
) -> int:
    """
    Replace nn.Linear modules in model with FP8Linear modules.

    Args:
        model: Model to modify
        weight_scales: Dict mapping layer names to weight scale tensors
        activation_scales: Dict mapping layer names to activation scale dicts
        use_fast: Use FP8LinearFast (True) or FP8LinearSlow (False)
        exclude_patterns: List of name patterns to exclude

    Returns:
        Number of replaced modules
    """
    if exclude_patterns is None:
        exclude_patterns = []

    LinearClass = FP8LinearFast if use_fast else FP8LinearSlow
    replaced_count = 0

    def should_exclude(name):
        for pattern in exclude_patterns:
            if pattern in name:
                return True
        return False

    # Build dict of modules to replace
    modules_to_replace = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and name in weight_scales:
            if not should_exclude(name):
                modules_to_replace[name] = module

    # Replace modules
    for name, module in modules_to_replace.items():
        # Get parent module and attr name
        parts = name.rsplit('.', 1)
        if len(parts) == 2:
            parent_name, attr_name = parts
            parent = dict(model.named_modules())[parent_name]
        else:
            parent = model
            attr_name = name

        # Get scales
        w_scale = weight_scales[name]['scale']
        a_scale = activation_scales.get(name, {}).get('scale', 1.0)

        # Create FP8 module
        fp8_module = LinearClass.from_linear(module, w_scale, a_scale)

        # Replace
        setattr(parent, attr_name, fp8_module)
        replaced_count += 1

    return replaced_count


# Testing utilities
def test_fp8_linear():
    """Test FP8Linear implementations"""
    print("Testing FP8Linear implementations...")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    in_features, out_features = 256, 128
    batch_size = 4

    # Create reference Linear
    linear = nn.Linear(in_features, out_features, bias=True).to(device).to(torch.bfloat16)

    # Create test input
    x = torch.randn(batch_size, in_features, device=device, dtype=torch.bfloat16)

    # Reference output
    with torch.no_grad():
        y_ref = linear(x)

    # Create FP8Linear (slow)
    weight_scale = linear.weight.abs().max(dim=1)[0] / FP8_E4M3_MAX + 1e-12
    fp8_slow = FP8LinearSlow.from_linear(linear, weight_scale)

    with torch.no_grad():
        y_slow = fp8_slow(x)

    # Compare
    mse_slow = ((y_ref - y_slow) ** 2).mean().item()
    max_diff_slow = (y_ref - y_slow).abs().max().item()
    print(f"  FP8LinearSlow vs Reference:")
    print(f"    MSE: {mse_slow:.6e}")
    print(f"    Max diff: {max_diff_slow:.6f}")

    # Create FP8Linear (fast)
    fp8_fast = FP8LinearFast.from_linear(linear, weight_scale)
    print(f"  FP8LinearFast scaled_mm available: {fp8_fast._scaled_mm_available}")

    with torch.no_grad():
        y_fast = fp8_fast(x)

    mse_fast = ((y_ref - y_fast) ** 2).mean().item()
    max_diff_fast = (y_ref - y_fast).abs().max().item()
    print(f"  FP8LinearFast vs Reference:")
    print(f"    MSE: {mse_fast:.6e}")
    print(f"    Max diff: {max_diff_fast:.6f}")

    print("Test complete!")


if __name__ == "__main__":
    test_fp8_linear()
