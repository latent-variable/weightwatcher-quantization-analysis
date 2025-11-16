#!/usr/bin/env python3
"""
WeightWatcher Quantization Analysis Script

Analyzes how different quantization levels affect the alpha metrics
of neural network layers using WeightWatcher framework.

Usage:
    python analyze_quantization.py --model meta-llama/Llama-3.2-3B --quantizations fp16 8bit 4bit
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import torch
import weightwatcher as ww
from transformers import AutoModelForCausalLM, AutoConfig

# Import quantization utilities
from quantization_utils import apply_quantization_to_model

# Check if running on Mac
IS_MAC = sys.platform == "darwin"
IS_MPS_AVAILABLE = torch.backends.mps.is_available() if IS_MAC else False


def get_device():
    """Determine the best device for the current platform."""
    if torch.cuda.is_available():
        return "cuda"
    elif IS_MPS_AVAILABLE:
        return "mps"
    else:
        return "cpu"


def clear_memory():
    """Clear GPU/MPS memory cache."""
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif IS_MPS_AVAILABLE:
        torch.mps.empty_cache()


def load_model_fp16(model_name: str, device: str = "auto"):
    """Load model in FP16 precision."""
    print(f"Loading {model_name} in FP16...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map=device,
        low_cpu_mem_usage=True,
    )
    print(f"  Model loaded successfully (FP16)")
    return model


def load_model_8bit(model_name: str, device: str = "auto"):
    """Load model in 8-bit quantization."""
    print(f"Loading {model_name} in 8-bit...")

    if IS_MAC:
        print("  Note: Using simulated 8-bit quantization (Mac-compatible)")
        print("  Weights are quantized to 8-bit precision but kept as float tensors")
        print("  This preserves compatibility with WeightWatcher analysis")

        # Load in FP16 first
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map=device,
            low_cpu_mem_usage=True,
        )

        # Apply simulated quantization
        # This quantizes weights to 8-bit precision but keeps them as float tensors
        # so WeightWatcher can still analyze them
        model = apply_quantization_to_model(model, bits=8, symmetric=True)
        print(f"  Model loaded and quantized to 8-bit (simulated quantization)")
        return model
    else:
        # Use bitsandbytes on CUDA
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                load_in_8bit=True,
                device_map=device,
                low_cpu_mem_usage=True,
            )
            print(f"  Model loaded successfully (8-bit via bitsandbytes)")
            return model
        except Exception as e:
            print(f"  Error loading with bitsandbytes: {e}")
            print("  Falling back to simulated 8-bit quantization")
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map=device,
                low_cpu_mem_usage=True,
            )
            model = apply_quantization_to_model(model, bits=8, symmetric=True)
            return model


def load_model_6bit(model_name: str, device: str = "auto"):
    """Load model in 6-bit quantization."""
    print(f"Loading {model_name} in 6-bit...")
    print("  Note: Using simulated 6-bit quantization")
    print("  Weights are quantized to 6-bit precision but kept as float tensors")

    # Load in FP16 first
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map=device,
        low_cpu_mem_usage=True,
    )

    # Apply simulated 6-bit quantization
    model = apply_quantization_to_model(model, bits=6, symmetric=True)
    print(f"  Model loaded and quantized to 6-bit (simulated quantization)")
    return model


def load_model_5bit(model_name: str, device: str = "auto"):
    """Load model in 5-bit quantization."""
    print(f"Loading {model_name} in 5-bit...")
    print("  Note: Using simulated 5-bit quantization")
    print("  Weights are quantized to 5-bit precision but kept as float tensors")

    # Load in FP16 first
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map=device,
        low_cpu_mem_usage=True,
    )

    # Apply simulated 5-bit quantization
    model = apply_quantization_to_model(model, bits=5, symmetric=True)
    print(f"  Model loaded and quantized to 5-bit (simulated quantization)")
    return model


def load_model_4bit(model_name: str, device: str = "auto"):
    """Load model in 4-bit quantization."""
    print(f"Loading {model_name} in 4-bit...")

    if IS_MAC:
        print("  Note: Using simulated 4-bit quantization (Mac-compatible)")
        print("  Weights are quantized to 4-bit precision but kept as float tensors")

        # Load in FP16 first
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map=device,
            low_cpu_mem_usage=True,
        )

        # Apply simulated 4-bit quantization
        model = apply_quantization_to_model(model, bits=4, symmetric=True)
        print(f"  Model loaded and quantized to 4-bit (simulated quantization)")
        return model
    else:
        try:
            from transformers import BitsAndBytesConfig

            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )

            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map=device,
                low_cpu_mem_usage=True,
            )
            print(f"  Model loaded successfully (4-bit NF4)")
            return model
        except Exception as e:
            print(f"  Error loading with 4-bit: {e}")
            print("  Falling back to simulated 4-bit quantization")
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map=device,
                low_cpu_mem_usage=True,
            )
            model = apply_quantization_to_model(model, bits=4, symmetric=True)
            return model


def load_model_2bit(model_name: str, device: str = "auto"):
    """
    Load model in 2-bit quantization.
    Note: This is experimental and uses simulated quantization.
    """
    print(f"Loading {model_name} in 2-bit...")
    print("  Note: Using simulated 2-bit quantization (experimental)")
    print("  Weights are quantized to 2-bit precision but kept as float tensors")
    print("  WARNING: Extreme quantization - expect significant quality degradation")

    # Load in FP16 first
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map=device,
        low_cpu_mem_usage=True,
    )

    # Apply simulated 2-bit quantization
    model = apply_quantization_to_model(model, bits=2, symmetric=True)
    print(f"  Model loaded and quantized to 2-bit (simulated quantization)")
    return model


QUANTIZATION_LOADERS = {
    "fp16": load_model_fp16,
    "8bit": load_model_8bit,
    "6bit": load_model_6bit,
    "5bit": load_model_5bit,
    "4bit": load_model_4bit,
    "2bit": load_model_2bit,
}


def analyze_model(model, model_name: str, quantization: str) -> pd.DataFrame:
    """Run WeightWatcher analysis on the model."""
    print(f"\nAnalyzing {quantization} model with WeightWatcher...")

    try:
        # Initialize WeightWatcher
        watcher = ww.WeightWatcher(model=model)

        # Run analysis (WeightWatcher automatically computes alpha)
        results = watcher.analyze()

        # Add metadata
        results['model_name'] = model_name
        results['quantization'] = quantization
        results['timestamp'] = datetime.now().isoformat()

        # Print summary
        print(f"  Analysis complete!")
        print(f"  Total layers analyzed: {len(results)}")
        print(f"  Alpha range: [{results['alpha'].min():.3f}, {results['alpha'].max():.3f}]")

        if 'alpha_weighted' in results.columns:
            alpha_hat = results['alpha_weighted'].mean()
            print(f"  Alpha-hat (weighted avg): {alpha_hat:.3f}")

        # Count layers outside optimal range
        outside_optimal = ((results['alpha'] < 2) | (results['alpha'] > 6)).sum()
        print(f"  Layers outside optimal range [2,6]: {outside_optimal}/{len(results)}")

        return results

    except Exception as e:
        print(f"  Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return None


def save_results(results: pd.DataFrame, output_dir: Path, quantization: str):
    """Save analysis results to CSV."""
    output_file = output_dir / f"results_{quantization}.csv"
    results.to_csv(output_file, index=False)
    print(f"  Results saved to: {output_file}")


def compare_quantizations(all_results: Dict[str, pd.DataFrame], output_dir: Path):
    """Generate comparison statistics across quantizations."""
    print("\n" + "="*80)
    print("QUANTIZATION COMPARISON SUMMARY")
    print("="*80)

    comparison = []

    for quant, results in all_results.items():
        if results is None:
            continue

        stats = {
            'quantization': quant,
            'num_layers': len(results),
            'alpha_mean': results['alpha'].mean(),
            'alpha_std': results['alpha'].std(),
            'alpha_min': results['alpha'].min(),
            'alpha_max': results['alpha'].max(),
            'alpha_median': results['alpha'].median(),
            'layers_below_2': (results['alpha'] < 2).sum(),
            'layers_above_6': (results['alpha'] > 6).sum(),
            'layers_optimal': ((results['alpha'] >= 2) & (results['alpha'] <= 6)).sum(),
        }

        if 'alpha_weighted' in results.columns:
            stats['alpha_hat'] = results['alpha_weighted'].mean()

        if 'log_norm' in results.columns:
            stats['log_norm_mean'] = results['log_norm'].mean()

        comparison.append(stats)

    comparison_df = pd.DataFrame(comparison)

    # Display comparison
    print("\nAlpha Statistics by Quantization:")
    print(comparison_df.to_string(index=False))

    # Save comparison
    comparison_file = output_dir / "quantization_comparison.csv"
    comparison_df.to_csv(comparison_file, index=False)
    print(f"\nComparison saved to: {comparison_file}")

    # Generate insights
    print("\n" + "="*80)
    print("KEY INSIGHTS")
    print("="*80)

    if len(comparison_df) > 1:
        baseline = comparison_df[comparison_df['quantization'] == 'fp16']
        if not baseline.empty:
            baseline_alpha = baseline['alpha_mean'].values[0]

            for _, row in comparison_df.iterrows():
                if row['quantization'] == 'fp16':
                    continue

                alpha_change = ((row['alpha_mean'] - baseline_alpha) / baseline_alpha) * 100
                optimal_pct = (row['layers_optimal'] / row['num_layers']) * 100

                print(f"\n{row['quantization'].upper()}:")
                print(f"  Alpha change from FP16: {alpha_change:+.2f}%")
                print(f"  Layers in optimal range: {optimal_pct:.1f}%")

                if 'alpha_hat' in row:
                    print(f"  Alpha-hat: {row['alpha_hat']:.3f}")

    return comparison_df


def main():
    parser = argparse.ArgumentParser(
        description="Analyze how quantization affects WeightWatcher alpha metrics"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="HuggingFace model name or path"
    )
    parser.add_argument(
        "--quantizations",
        nargs="+",
        default=["fp16", "8bit"],
        choices=["fp16", "8bit", "6bit", "5bit", "4bit", "2bit"],
        help="Quantization levels to test"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="../results/metrics",
        help="Output directory for results"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use (auto, cpu, cuda, mps)"
    )

    args = parser.parse_args()

    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # System info
    print("="*80)
    print("WEIGHTWATCHER QUANTIZATION ANALYSIS")
    print("="*80)
    print(f"Model: {args.model}")
    print(f"Quantizations: {', '.join(args.quantizations)}")
    print(f"Platform: {sys.platform}")
    print(f"PyTorch: {torch.__version__}")
    print(f"WeightWatcher: {ww.__version__}")

    device = get_device() if args.device == "auto" else args.device
    print(f"Device: {device}")
    print("="*80)

    # Run analysis for each quantization
    all_results = {}

    for quant in args.quantizations:
        print(f"\n{'='*80}")
        print(f"PROCESSING: {quant.upper()}")
        print(f"{'='*80}")

        try:
            # Load model
            loader = QUANTIZATION_LOADERS[quant]
            model = loader(args.model, device=device)

            # Analyze
            results = analyze_model(model, args.model, quant)

            if results is not None:
                all_results[quant] = results
                save_results(results, output_dir, quant)

            # Clean up
            del model
            clear_memory()

        except Exception as e:
            print(f"\nError processing {quant}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Generate comparison
    if all_results:
        compare_quantizations(all_results, output_dir)

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"Results saved to: {output_dir}")
    print("\nNext steps:")
    print("  1. Run visualize_results.py to generate plots")
    print("  2. Review the quantization_comparison.csv file")
    print("  3. Examine individual layer metrics in results_*.csv files")


if __name__ == "__main__":
    main()
