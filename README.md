# WeightWatcher Quantization Analysis

This project analyzes how quantization affects the layer quality metrics (alpha) in neural networks using the WeightWatcher framework. We test Qwen2.5-0.5B-Instruct across six quantization levels (FP16, 8-bit, 6-bit, 5-bit, 4-bit, 2-bit) to understand how compression impacts the Heavy-Tailed Self-Regularization (HTSR) properties of neural networks.

## 🎯 Key Results

Tested on **Qwen2.5-0.5B-Instruct** (168 layers, 24 transformer blocks):

| Quantization | Alpha Mean | Change from FP16 | Correlation | Layers in Optimal Range [2,6] |
|--------------|------------|------------------|-------------|-------------------------------|
| **FP16** (baseline) | 6.312 | — | 1.000 | 92/168 (54.8%) |
| **8-bit** | 6.294 | -0.30% ✅ | 0.988 | 93/168 (55.4%) |
| **6-bit** | 6.352 | +0.63% ✅ | 0.940 | 91/168 (54.2%) |
| **5-bit** | 6.926 | +9.72% ⚠️ | 0.775 | 83/168 (49.4%) |
| **4-bit** | 7.738 | +22.58% ⚠️ | 0.591 | 66/168 (39.3%) |
| **2-bit** | 2.783 | -55.91% 💀 | 0.238 | 112/168 (66.7%)* |

\* *2-bit appears to have high optimal percentage, but 48 layers have α < 2 (severe corruption)*

### Key Findings

1. **8-bit and 6-bit are essentially lossless**: <1% change in alpha, >0.94 correlation
2. **5-bit shows measurable degradation**: +9.72% alpha increase, quality starts degrading
3. **4-bit has significant quality loss**: +22.58% alpha increase, only 39% layers optimal
4. **2-bit is catastrophically degraded**: 48 layers corrupted (α < 2), including negative alpha values

**Conclusion**: For production deployment, **6-bit or 8-bit quantization is recommended** as they preserve model quality according to Heavy-Tailed Self-Regularization metrics.

### Quantization Method

This analysis uses **naive symmetric linear quantization** as a conservative baseline:
- Simple uniform quantization: `q = round(w / scale)`
- Single scale per weight matrix
- No outlier handling or grouping

Production quantization methods (GGUF K-quants, MLX group quantization) use more sophisticated techniques and would likely show **less degradation** than reported here. These results represent a **conservative lower bound** on quantization quality.

## Overview

WeightWatcher is a diagnostic tool that analyzes Deep Neural Networks without requiring training or test data. It computes the **alpha (α)** metric for each layer by analyzing the spectral properties of weight matrices. The alpha metric:

- Should typically be between 2 and 6 for well-trained layers
- Correlates with layer quality and model generalization
- Can predict test accuracy without access to test data
- Is based on Heavy-Tailed Random Matrix Theory (HTRMT)

### Research Question

**How does quantization (fp16, 8-bit, 4-bit, 2-bit) affect the alpha metric and overall model quality as measured by WeightWatcher?**

This analysis helps us understand:
- Whether quantized models maintain their heavy-tailed properties
- At what quantization level the model quality degrades significantly
- How compression affects layer-wise spectral properties

## Background

This project is based on the Semi-Empirical Theory of Learning (SETOL) described in the paper "SETOL: A Semi-Empirical Theory of (Deep) Learning" (arXiv:2507.17912v2). The theory provides a formal explanation of:

- **Alpha (α)**: Power law exponent of layer weight eigenvalue distributions
- **Alpha-hat (α̂)**: Weighted average alpha across all layers
- **ERG metric**: Exact Renormalization Group quality measure

These metrics enable data-free quality assessment of pretrained models.

## System Requirements

- **Platform**: macOS (Apple Silicon recommended for optimal performance)
- **Python**: 3.8 or higher
- **RAM**: 16GB minimum (32GB recommended for larger quantizations)
- **Storage**: ~20GB for model files and cache

## Installation

### 1. Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install --upgrade pip
pip install weightwatcher
pip install torch torchvision torchaudio
pip install transformers accelerate
pip install bitsandbytes  # For 8-bit and 4-bit quantization
pip install pandas matplotlib seaborn jupyter
```

### 3. Verify Installation

```bash
python -c "import weightwatcher as ww; print(f'WeightWatcher version: {ww.__version__}')"
```

## Project Structure

```
weightwatcher/
├── README.md                          # This file
├── docs/
│   └── 2507.17912v2.txt              # SETOL theory paper
├── scripts/
│   ├── analyze_quantization.py       # Main analysis script
│   ├── load_model.py                 # Model loading utilities
│   └── visualize_results.py          # Plotting and visualization
├── results/
│   ├── metrics/                      # CSV files with alpha metrics
│   ├── plots/                        # Visualization outputs
│   └── reports/                      # Analysis summaries
└── notebooks/
    └── quantization_analysis.ipynb   # Interactive analysis
```

## Usage

### Quick Start

```python
import torch
from transformers import AutoModelForCausalLM
import weightwatcher as ww

# Load model in different quantizations
model_fp16 = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-4B-Instruct-2507",
    torch_dtype=torch.float16,
    device_map="auto"
)

# Analyze with WeightWatcher
watcher = ww.WeightWatcher(model=model_fp16)
results_fp16 = watcher.analyze()

# View alpha metrics
print(results_fp16[['layer_id', 'alpha', 'alpha_weighted']])
```

### Full Analysis Pipeline

```bash
# Run complete quantization analysis
python scripts/analyze_quantization.py \
    --model "Qwen/Qwen3-4B-Instruct-2507" \
    --quantizations fp16 8bit 4bit 2bit \
    --output-dir results/

# Generate visualization
python scripts/visualize_results.py --results-dir results/
```

## Quantization Configurations

### FP16 (Baseline)
- **Precision**: 16-bit floating point
- **Use case**: Baseline for comparison
- **Memory**: ~6GB

### 8-bit Quantization
- **Method**: LLM.int8() via bitsandbytes
- **Precision**: 8-bit integers with outlier handling
- **Memory**: ~3GB
- **Expected impact**: Minimal quality loss

### 4-bit Quantization
- **Method**: QLoRA-style 4-bit NormalFloat
- **Precision**: 4-bit with double quantization
- **Memory**: ~1.5GB
- **Expected impact**: Moderate quality impact

### 2-bit Quantization
- **Method**: Extreme compression
- **Precision**: 2-bit representation
- **Memory**: ~0.75GB
- **Expected impact**: Significant quality degradation

## Metrics to Analyze

For each quantization level, we examine:

1. **Layer-wise Alpha (α)**
   - Distribution across layers
   - Layers outside optimal range [2, 6]
   - Changes from baseline (fp16)

2. **Alpha-hat (α̂)**
   - Weighted average across all layers
   - Overall model quality indicator

3. **Spectral Properties**
   - Log spectral norm
   - Frobenius norm
   - Stable rank

4. **Power Law Fit Quality**
   - R² of power law fits
   - Presence of heavy-tailed distributions

## Expected Results

### Hypothesis

As quantization bit-width decreases:
- Alpha values may shift outside optimal [2, 6] range
- Alpha-hat will decrease, indicating quality loss
- Spectral norms will change due to weight compression
- Power law fits may degrade at extreme quantization (2-bit)

### Interpretation Guide

**Alpha in optimal range [2, 6]**: Layer maintains good generalization properties

**Alpha > 6**: Layer may be undertrained or poorly regularized (common in aggressive quantization)

**Alpha < 2**: Layer may be over-regularized or corrupted

**Alpha-hat trend**: Decreasing alpha-hat across quantization levels indicates progressive quality loss

## Running on Mac

### Quantization Approach on Mac

Since bitsandbytes (used for efficient 8/4-bit quantization) requires CUDA, we use **simulated quantization** on Mac:

1. Load model in FP16 with MPS acceleration
2. Apply symmetric quantization to all Linear layer weights
3. Quantize to target bit-width → Dequantize back to float16
4. Analyze with WeightWatcher

**This approach:**
- ✅ Preserves the precision loss effects of quantization
- ✅ Works with WeightWatcher's eigenvalue analysis
- ✅ Compatible with MPS/CPU on Mac
- ✅ Supports all bit-widths: 8-bit, 4-bit, 2-bit
- ❌ Not optimized for inference (but we're analyzing weights, not running the model!)

### Apple Silicon Optimization

```python
# Use MPS (Metal Performance Shaders) when available
device = "mps" if torch.backends.mps.is_available() else "cpu"

# Load in FP16 (works great with MPS)
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-4B-Instruct-2507",
    torch_dtype=torch.float16,
    device_map=device
)

# Apply simulated quantization (keeps weights as float for analysis)
from scripts.quantization_utils import apply_quantization_to_model
model = apply_quantization_to_model(model, bits=8, symmetric=True)
```

### Memory Management

```python
# Clear cache between runs
import gc
torch.mps.empty_cache() if torch.backends.mps.is_available() else torch.cuda.empty_cache()
gc.collect()
```

## Visualization

Generate comparative plots:

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Compare alpha distributions
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
for idx, quant in enumerate(['fp16', '8bit', '4bit', '2bit']):
    ax = axes[idx // 2, idx % 2]
    results[quant]['alpha'].hist(ax=ax, bins=20)
    ax.set_title(f'Alpha Distribution - {quant}')
    ax.set_xlabel('Alpha')
    ax.axvspan(2, 6, alpha=0.3, color='green', label='Optimal Range')
```

## Troubleshooting

### Issue: Model download fails
**Solution**: Ensure you have HuggingFace access token for Llama models:
```bash
huggingface-cli login
```

### Issue: Out of memory on Mac
**Solution**: Close other applications or test with a smaller model:
```bash
./run_analysis.sh --model meta-llama/Llama-3.2-1B
```

### Issue: "torch.nn.Linear has no attribute weight after quantization"
**Solution**: This shouldn't happen with the updated simulated quantization approach. If it does:
- Ensure you're using `quantization_utils.py` (not `torch.quantization.quantize_dynamic`)
- Check that the model loads correctly in FP16 first

### Issue: Quantization results identical to FP16
**Solution**: Verify quantization is applied:
```python
# After loading model
print(model.model.layers[0].self_attn.q_proj.weight.unique().numel())
# Should be much smaller for quantized models (e.g., 256 for 8-bit)
```

### Note on bitsandbytes
**Mac users**: The framework automatically uses simulated quantization instead of bitsandbytes. This:
- Works on Mac without CUDA
- Produces analyzable results for WeightWatcher
- Shows quantization effects on alpha metrics
- Isn't optimized for inference (but that's not our goal!)

## References

1. **WeightWatcher Framework**: https://weightwatcher.ai/
2. **SETOL Paper**: Martin, C. H., & Hinrichs, C. (2025). "SETOL: A Semi-Empirical Theory of (Deep) Learning." arXiv:2507.17912v2
3. **Heavy-Tailed Self-Regularization**: Theory explaining why deep learning generalizes
4. **Llama 3.2**: Meta's 3B parameter language model

## Citation

If you use this analysis in your research, please cite:

```bibtex
@article{martin2025setol,
  title={SETOL: A Semi-Empirical Theory of (Deep) Learning},
  author={Martin, Charles H. and Hinrichs, Christopher},
  journal={arXiv preprint arXiv:2507.17912},
  year={2025}
}

@software{weightwatcher,
  title={WeightWatcher: A Diagnostic Tool for Deep Neural Networks},
  url={https://weightwatcher.ai/}
}
```

## License

This project is for research and educational purposes. Please respect the licenses of:
- WeightWatcher (Apache 2.0)
- Llama 3.2 (Meta's Community License)
- Transformers library (Apache 2.0)

## Contributing

Contributions welcome! Areas of interest:
- Support for additional models (GPT, BERT, Vision Transformers)
- Enhanced visualization dashboards
- Statistical significance testing
- Correlation with actual task performance

## Contact

For questions about:
- **WeightWatcher**: https://github.com/CalculatedContent/WeightWatcher
- **This project**: [Your contact information]

---

**Note**: This analysis requires significant computational resources. Start with fp16 and 8-bit before attempting 4-bit and 2-bit quantization.
