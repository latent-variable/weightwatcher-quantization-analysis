# Quick Start Guide

Get started with WeightWatcher quantization analysis in 5 minutes!

## Prerequisites

- Python 3.8 or higher
- macOS (Apple Silicon recommended) or Linux with CUDA
- ~20GB free disk space
- HuggingFace account (for Llama model access)

## Step 1: Setup

Run the setup script:

```bash
cd /Users/linovaldovinos/Documents/LatentPlayground/weightwatcher
./setup.sh
```

This will:
- Create a virtual environment
- Install all dependencies
- Verify installations

## Step 2: HuggingFace Authentication (Optional)

For most models like Qwen, you don't need authentication. But if you want to use gated models later:

```bash
source venv/bin/activate
huggingface-cli login
```

Enter your HuggingFace token when prompted. Get a token at: https://huggingface.co/settings/tokens

**Note:** Qwen3-4B-Instruct-2507 (the default model) doesn't require authentication!

## Step 3: Run Analysis

### Option A: Automated Script (Recommended)

Run everything with one command:

```bash
./run_analysis.sh
```

This will:
- Load Qwen3 4B in FP16 and 8-bit
- Run WeightWatcher analysis on both
- Generate comparison plots
- Create a detailed report

**For more quantizations:**

```bash
./run_analysis.sh --quantizations fp16 8bit 4bit
```

**Note**: 4-bit and 2-bit require CUDA. On Mac, only fp16 and 8-bit are fully supported.

### Option B: Manual Step-by-Step

```bash
source venv/bin/activate
cd scripts

# 1. Run analysis
python analyze_quantization.py --model Qwen/Qwen3-4B-Instruct-2507 --quantizations fp16 8bit

# 2. Generate visualizations
python visualize_results.py --results-dir ../results/metrics
```

### Option C: Interactive Jupyter Notebook

```bash
source venv/bin/activate
jupyter notebook notebooks/quantization_analysis.ipynb
```

Run cells sequentially to:
- Load models interactively
- Analyze with WeightWatcher
- Generate custom visualizations
- Explore results

## Step 4: View Results

### Metrics (CSV files)
```bash
ls results/metrics/
# results_fp16.csv, results_8bit.csv, etc.
```

### Visualizations (PNG files)
```bash
ls results/plots/
# alpha_distributions.png
# alpha_comparison_boxplot.png
# layerwise_alpha_comparison.png
# alpha_statistics_summary.png
```

### Analysis Report (Text summary)
```bash
cat results/analysis_report.txt
```

## Understanding the Results

### Alpha (α) Metric

The key metric is **alpha (α)**, which measures the power-law exponent of layer weight distributions:

- **α ∈ [2, 6]**: ✅ Well-trained layer
- **α > 6**: ⚠️ Undertrained or poorly regularized
- **α < 2**: ⚠️ Over-regularized or corrupted

### What to Look For

1. **Mean Alpha Change**: How does quantization affect average alpha?
   - Small change (<5%): Quantization preserves layer quality
   - Large change (>10%): Significant quality degradation

2. **Layers Outside Optimal Range**: How many layers fall outside [2, 6]?
   - More layers outside range = worse quality

3. **Distribution Shift**: Does the alpha distribution change shape?
   - Similar distribution = stable quantization
   - Different distribution = structural changes

4. **Layer-wise Patterns**: Which layers are most affected?
   - Early layers vs late layers
   - Attention vs feedforward

## Mac-Specific Notes

### Supported Quantizations

- ✅ **FP16**: Full support via PyTorch with MPS acceleration
- ✅ **8-bit**: Simulated quantization (weights quantized to 8-bit precision, kept as float tensors for WeightWatcher compatibility)
- ✅ **4-bit**: Simulated quantization (same approach, works on Mac!)
- ✅ **2-bit**: Simulated quantization (experimental, expect significant degradation)

### How Quantization Works on Mac

**On Mac**, we use **simulated quantization**:
1. Load model in FP16 with MPS acceleration
2. Quantize weights to target bit-width (8/4/2 bit)
3. Immediately dequantize back to float16
4. This preserves quantization **precision loss** while keeping tensors WeightWatcher-compatible

**On CUDA**, we use **actual quantization** (bitsandbytes):
- True 8-bit/4-bit inference with optimized kernels
- More memory efficient
- Slightly different quantization scheme (LLM.int8(), NF4)

**Why simulated quantization?**
- ✅ Works with WeightWatcher's eigenvalue analysis
- ✅ Shows real quantization effects on alpha metrics
- ✅ Compatible with MPS acceleration for model loading
- ✅ Accurately represents precision loss from quantization
- ❌ Not optimized for inference (but we're analyzing, not running inference!)

### Performance Tips

1. **MPS acceleration** is used for FP16 model loading (automatic)

2. **Monitor memory** with Activity Monitor during analysis

3. **Close other applications** to free up RAM

4. **All quantization levels work on Mac** - no need for CUDA!

## Common Issues

### Issue: "Out of memory"
**Solution**: Reduce model size or use smaller model:
```bash
./run_analysis.sh --model meta-llama/Llama-3.2-1B
```

### Issue: "bitsandbytes not working"
**Solution**: On Mac, this is expected. The script automatically falls back to `torch.quantization`.

### Issue: "Cannot access model"
**Solution**:
- Qwen models don't require authentication
- If using a gated model (like Llama), login to HuggingFace:
```bash
huggingface-cli login
```

### Issue: "Module not found"
**Solution**: Ensure virtual environment is activated:
```bash
source venv/bin/activate
```

## Next Steps

### Experiment with Different Models

```bash
./run_analysis.sh --model microsoft/phi-2
./run_analysis.sh --model google/gemma-2b
```

### Customize Analysis

Edit `scripts/analyze_quantization.py` to:
- Change WeightWatcher parameters
- Add custom metrics
- Modify quantization configs

### Compare with Task Performance

After WeightWatcher analysis, benchmark models on actual tasks to see if alpha degradation correlates with performance loss.

## Need Help?

- **WeightWatcher docs**: https://weightwatcher.ai/
- **Research paper**: `docs/2507.17912v2.txt`
- **GitHub issues**: https://github.com/CalculatedContent/WeightWatcher/issues

## Example Workflow

Here's a complete example workflow:

```bash
# 1. Setup (one-time)
./setup.sh
source venv/bin/activate

# 2. Run analysis (no login needed for Qwen!)
./run_analysis.sh --quantizations fp16 8bit

# 3. View report
cat results/analysis_report.txt

# 4. Open plots
open results/plots/alpha_statistics_summary.png

# 5. Explore interactively
jupyter notebook notebooks/quantization_analysis.ipynb
```

That's it! You're now analyzing how quantization affects model quality using WeightWatcher.
