#!/usr/bin/env python3
"""
Quick WeightWatcher test - analyzes only first few layers for speed.

Usage:
    python quick_test.py --model Qwen/Qwen3-4B-Instruct-2507 --num-layers 5
"""

import argparse
import sys
import torch
from transformers import AutoModelForCausalLM
import weightwatcher as ww
import pandas as pd
from pathlib import Path
from quantization_utils import apply_quantization_to_model

IS_MAC = sys.platform == "darwin"
IS_MPS_AVAILABLE = torch.backends.mps.is_available() if IS_MAC else False


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif IS_MPS_AVAILABLE:
        return "mps"
    return "cpu"


def analyze_subset_of_layers(model, quantization: str, max_layers: int = 5):
    """Analyze only first N layers for speed."""
    print(f"\nAnalyzing {quantization} (first {max_layers} layers only)...")
    print(f"  Initializing WeightWatcher...")

    # Enable verbose output
    import logging
    logging.basicConfig(level=logging.INFO)

    watcher = ww.WeightWatcher(model=model)

    print(f"  Running analysis (this may take a few minutes per layer)...")
    print(f"  Computing eigenvalues for weight matrices...")

    # Run with timing
    import time
    start = time.time()
    results = watcher.analyze()
    elapsed = time.time() - start

    print(f"  ✓ Analysis complete in {elapsed:.1f}s")

    # Keep only first N layers
    results_subset = results.head(max_layers).copy()
    results_subset['quantization'] = quantization

    print(f"  Analyzed {len(results_subset)} layers")
    print(f"  Alpha range: [{results_subset['alpha'].min():.3f}, {results_subset['alpha'].max():.3f}]")
    print(f"  Alpha mean: {results_subset['alpha'].mean():.3f}")

    return results_subset


def main():
    parser = argparse.ArgumentParser(description="Quick WeightWatcher test")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--num-layers", type=int, default=5, help="Number of layers to analyze")
    parser.add_argument("--quantizations", nargs="+", default=["fp16", "8bit", "4bit"])

    args = parser.parse_args()

    device = get_device()
    print("="*80)
    print("QUICK WEIGHTWATCHER TEST")
    print("="*80)
    print(f"Model: {args.model}")
    print(f"Analyzing first {args.num_layers} layers only")
    print(f"Quantizations: {', '.join(args.quantizations)}")
    print(f"Device: {device}")
    print("="*80)

    all_results = []

    for quant in args.quantizations:
        print(f"\n{'='*80}")
        print(f"Loading {quant.upper()} model...")
        print(f"{'='*80}")

        # Load base model
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.float16,
            device_map=device,
            low_cpu_mem_usage=True
        )

        # Apply quantization if needed
        if quant == "8bit":
            model = apply_quantization_to_model(model, bits=8, symmetric=True)
        elif quant == "4bit":
            model = apply_quantization_to_model(model, bits=4, symmetric=True)
        elif quant == "2bit":
            model = apply_quantization_to_model(model, bits=2, symmetric=True)

        # Analyze subset
        results = analyze_subset_of_layers(model, quant, args.num_layers)
        all_results.append(results)

        # Cleanup
        del model
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif IS_MPS_AVAILABLE:
            torch.mps.empty_cache()

    # Combine and save
    combined = pd.concat(all_results, ignore_index=True)
    output_file = Path("../results/quick_test_results.csv")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_file, index=False)

    print("\n" + "="*80)
    print("QUICK COMPARISON")
    print("="*80)

    for quant in args.quantizations:
        subset = combined[combined['quantization'] == quant]
        print(f"\n{quant.upper()}:")
        print(f"  Mean alpha: {subset['alpha'].mean():.3f}")
        print(f"  Std alpha:  {subset['alpha'].std():.3f}")

    print(f"\n\nResults saved to: {output_file}")
    print("This was a QUICK TEST with limited layers.")
    print("Run the full analysis for complete results.")


if __name__ == "__main__":
    main()
