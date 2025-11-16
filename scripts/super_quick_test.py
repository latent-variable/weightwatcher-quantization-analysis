#!/usr/bin/env python3
"""
Super quick test with a tiny model (30 seconds total).

Uses a small model to verify the quantization analysis framework works.
"""

import sys
import torch
from transformers import AutoModelForCausalLM
import weightwatcher as ww
import pandas as pd
from pathlib import Path
from quantization_utils import apply_quantization_to_model
import time

IS_MAC = sys.platform == "darwin"
IS_MPS_AVAILABLE = torch.backends.mps.is_available() if IS_MAC else False


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif IS_MPS_AVAILABLE:
        return "mps"
    return "cpu"


def main():
    # Use a tiny model for speed
    MODEL = "Qwen/Qwen2.5-0.5B-Instruct"  # Only 0.5B params, 24 layers

    device = get_device()
    print("="*80)
    print("SUPER QUICK WEIGHTWATCHER TEST")
    print("="*80)
    print(f"Model: {MODEL} (tiny model for speed)")
    print(f"Testing: FP16 vs 4-bit quantization")
    print(f"Analyzing: First 3 layers only")
    print(f"Device: {device}")
    print(f"Expected time: ~30-60 seconds")
    print("="*80)

    all_results = []

    for quant, bits in [("fp16", None), ("4bit", 4)]:
        print(f"\n{'='*80}")
        print(f"Testing {quant.upper()}")
        print(f"{'='*80}")

        # Load model
        print(f"  Loading model...")
        start = time.time()
        model = AutoModelForCausalLM.from_pretrained(
            MODEL,
            torch_dtype=torch.float16,
            device_map=device,
            low_cpu_mem_usage=True
        )
        load_time = time.time() - start
        print(f"  ✓ Loaded in {load_time:.1f}s")

        # Apply quantization
        if bits:
            print(f"  Applying {bits}-bit quantization...")
            start = time.time()
            model = apply_quantization_to_model(model, bits=bits, symmetric=True)
            quant_time = time.time() - start
            print(f"  ✓ Quantized in {quant_time:.1f}s")

        # Analyze
        print(f"  Analyzing with WeightWatcher...")
        start = time.time()
        watcher = ww.WeightWatcher(model=model)
        results = watcher.analyze()
        analyze_time = time.time() - start
        print(f"  ✓ Analyzed {len(results)} layers in {analyze_time:.1f}s")

        # Keep first 3 layers
        results_subset = results.head(3).copy()
        results_subset['quantization'] = quant

        print(f"\n  Results (first 3 layers):")
        print(f"    Alpha mean: {results_subset['alpha'].mean():.3f}")
        print(f"    Alpha std:  {results_subset['alpha'].std():.3f}")
        print(f"    Alpha range: [{results_subset['alpha'].min():.3f}, {results_subset['alpha'].max():.3f}]")

        all_results.append(results_subset)

        # Cleanup
        del model, watcher
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif IS_MPS_AVAILABLE:
            torch.mps.empty_cache()

    # Compare
    print("\n" + "="*80)
    print("COMPARISON")
    print("="*80)

    combined = pd.concat(all_results, ignore_index=True)

    fp16_mean = combined[combined['quantization'] == 'fp16']['alpha'].mean()
    q4bit_mean = combined[combined['quantization'] == '4bit']['alpha'].mean()

    change = q4bit_mean - fp16_mean
    pct_change = (change / fp16_mean) * 100

    print(f"\nFP16 mean alpha:  {fp16_mean:.3f}")
    print(f"4-bit mean alpha: {q4bit_mean:.3f}")
    print(f"Change:           {change:+.3f} ({pct_change:+.2f}%)")

    if abs(pct_change) > 5:
        print(f"\n✓ Quantization has a measurable effect on alpha!")
    else:
        print(f"\n✓ Alpha remains stable despite quantization")

    # Save results
    output_file = Path("../results/super_quick_test.csv")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_file, index=False)

    print(f"\n✓ Results saved to: {output_file}")
    print("\n" + "="*80)
    print("Framework verified! You can now run full analysis with larger models.")
    print("="*80)


if __name__ == "__main__":
    main()
