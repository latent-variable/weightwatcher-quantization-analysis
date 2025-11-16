# Model Updated: Qwen3-4B-Instruct-2507

All scripts and documentation have been updated to use **Qwen/Qwen3-4B-Instruct-2507** as the default model instead of Meta's Llama 3.2 3B.

## Why Qwen?

- ✅ **No gating** - No authentication required
- ✅ **No Meta** - Avoid their restrictive licensing
- ✅ **Similar size** - 4B parameters vs 3B (comparable for analysis)
- ✅ **Great architecture** - 36 layers, 2560 hidden size
- ✅ **Well-trained** - High quality model from Alibaba Cloud

## What Was Updated

### Scripts
- ✅ `scripts/analyze_quantization.py` - Default model changed
- ✅ `run_analysis.sh` - Default model and help text updated
- ✅ `notebooks/quantization_analysis.ipynb` - Model name and title updated

### Documentation
- ✅ `README.md` - All examples updated to use Qwen
- ✅ `QUICKSTART.md` - Authentication now optional, examples updated
- ✅ `run_analysis.sh --help` - Shows Qwen as default

## Quick Commands

### Run with default model (Qwen3-4B):
```bash
./run_analysis.sh
```

### Run with all quantizations:
```bash
./run_analysis.sh --quantizations fp16 8bit 4bit 2bit
```

### Run with a different model:
```bash
./run_analysis.sh --model Qwen/Qwen2.5-3B-Instruct
./run_analysis.sh --model microsoft/phi-2
```

## Model Comparison

| Model | Size | Gating | Authentication | Notes |
|-------|------|--------|----------------|-------|
| **Qwen3-4B-Instruct-2507** | 4B | ❌ No | Not required | **Default, recommended** |
| Qwen2.5-3B-Instruct | 3B | ❌ No | Not required | Smaller, faster |
| Meta Llama 3.2 3B | 3B | ✅ Yes | Required | Gated by Meta |
| Phi-2 | 2.7B | ❌ No | Not required | Microsoft, smaller |

## Expected Analysis Results

Since Qwen3-4B is a well-trained model, you should see:

- **FP16**: Most layers with alpha ∈ [2, 6]
- **8-bit**: Minimal degradation (~1-3% alpha change)
- **4-bit**: Moderate degradation (~5-10% alpha change)
- **2-bit**: Significant degradation (>15% alpha change)

This is a great model for testing the WeightWatcher quantization analysis framework!

## No Authentication Needed!

Unlike Llama, you can start analyzing immediately:

```bash
cd /Users/linovaldovinos/Documents/LatentPlayground/weightwatcher
source venv/bin/activate
./run_analysis.sh
```

No HuggingFace login, no access requests, no waiting. Just pure analysis! 🎉
