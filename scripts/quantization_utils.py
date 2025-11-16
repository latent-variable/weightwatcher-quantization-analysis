#!/usr/bin/env python3
"""
Quantization utilities for Mac-compatible weight analysis.

This provides a way to simulate quantization effects while keeping
weights in a format that WeightWatcher can analyze.
"""

import torch
import torch.nn as nn
from typing import Literal


def quantize_weight_tensor(
    weight: torch.Tensor,
    bits: Literal[2, 4, 5, 6, 8, 16],
    symmetric: bool = True
) -> torch.Tensor:
    """
    Simulate quantization by quantizing and dequantizing a weight tensor.

    This keeps the result as a float tensor that WeightWatcher can analyze,
    but with the precision degradation that would occur from quantization.

    Args:
        weight: Original weight tensor (float)
        bits: Number of bits for quantization (2, 4, 5, 6, 8, or 16)
        symmetric: Use symmetric quantization (range is -max to +max)

    Returns:
        Dequantized weight tensor (float, but with quantization artifacts)
    """
    if bits == 16:
        # FP16 - just convert dtype
        return weight.to(torch.float16).to(weight.dtype)

    # Compute quantization parameters
    if symmetric:
        # Symmetric: range is -abs_max to +abs_max
        abs_max = weight.abs().max()
        q_min = -(2 ** (bits - 1))
        q_max = 2 ** (bits - 1) - 1
        scale = abs_max / q_max
        zero_point = 0
    else:
        # Asymmetric: range is min to max
        w_min = weight.min()
        w_max = weight.max()
        q_min = 0
        q_max = 2 ** bits - 1
        scale = (w_max - w_min) / (q_max - q_min)
        zero_point = q_min - w_min / scale

    # Avoid division by zero
    if scale == 0:
        return weight

    # Quantize: float -> int
    quantized = torch.clamp(
        torch.round(weight / scale + zero_point),
        q_min,
        q_max
    )

    # Dequantize: int -> float
    dequantized = (quantized - zero_point) * scale

    return dequantized.to(weight.dtype)


def apply_quantization_to_model(
    model: nn.Module,
    bits: Literal[2, 4, 5, 6, 8, 16],
    symmetric: bool = True,
    layer_types: tuple = (nn.Linear,)
) -> nn.Module:
    """
    Apply simulated quantization to all weights in specified layer types.

    This modifies the model in-place but keeps all weights as regular
    float tensors, making them compatible with WeightWatcher analysis.

    Args:
        model: PyTorch model
        bits: Number of bits for quantization
        symmetric: Use symmetric quantization
        layer_types: Tuple of layer types to quantize (default: Linear only)

    Returns:
        Modified model (same object, modified in-place)
    """
    with torch.no_grad():
        for name, module in model.named_modules():
            if isinstance(module, layer_types):
                if hasattr(module, 'weight') and module.weight is not None:
                    # Quantize the weight
                    original_device = module.weight.device
                    original_dtype = module.weight.dtype

                    quantized_weight = quantize_weight_tensor(
                        module.weight.data,
                        bits=bits,
                        symmetric=symmetric
                    )

                    module.weight.data = quantized_weight.to(
                        device=original_device,
                        dtype=original_dtype
                    )

                # Optionally quantize bias too
                if hasattr(module, 'bias') and module.bias is not None:
                    # Usually bias is kept in higher precision
                    # but we can quantize it too for consistency
                    original_device = module.bias.device
                    original_dtype = module.bias.dtype

                    quantized_bias = quantize_weight_tensor(
                        module.bias.data,
                        bits=bits,
                        symmetric=symmetric
                    )

                    module.bias.data = quantized_bias.to(
                        device=original_device,
                        dtype=original_dtype
                    )

    return model


def estimate_quantization_error(
    original_weight: torch.Tensor,
    quantized_weight: torch.Tensor
) -> dict:
    """
    Calculate quantization error metrics.

    Returns:
        Dictionary with error metrics (MSE, MAE, SNR, etc.)
    """
    mse = torch.mean((original_weight - quantized_weight) ** 2).item()
    mae = torch.mean(torch.abs(original_weight - quantized_weight)).item()

    # Signal-to-noise ratio
    signal_power = torch.mean(original_weight ** 2).item()
    noise_power = mse
    snr = 10 * torch.log10(torch.tensor(signal_power / (noise_power + 1e-10))).item()

    # Relative error
    relative_error = (mae / (torch.abs(original_weight).mean().item() + 1e-10))

    return {
        'mse': mse,
        'mae': mae,
        'snr_db': snr,
        'relative_error': relative_error
    }


if __name__ == "__main__":
    # Test quantization
    print("Testing quantization utilities...")

    # Create a sample weight tensor
    torch.manual_seed(42)
    weight = torch.randn(512, 512)

    print(f"\nOriginal weight stats:")
    print(f"  Mean: {weight.mean():.6f}")
    print(f"  Std:  {weight.std():.6f}")
    print(f"  Min:  {weight.min():.6f}")
    print(f"  Max:  {weight.max():.6f}")

    # Test different bit widths
    for bits in [8, 6, 5, 4, 2]:
        quantized = quantize_weight_tensor(weight, bits=bits)
        errors = estimate_quantization_error(weight, quantized)

        print(f"\n{bits}-bit quantization:")
        print(f"  Unique values: {len(torch.unique(quantized))}")
        print(f"  MSE:  {errors['mse']:.6f}")
        print(f"  MAE:  {errors['mae']:.6f}")
        print(f"  SNR:  {errors['snr_db']:.2f} dB")
        print(f"  Rel Error: {errors['relative_error']:.6f}")
