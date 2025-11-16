#!/bin/bash
# Setup script for WeightWatcher Quantization Analysis

set -e  # Exit on error

echo "=================================="
echo "WeightWatcher Quantization Setup"
echo "=================================="
echo ""

# Check Python version
echo "Checking Python version..."
python3 --version

# Create virtual environment
echo ""
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install -r requirements.txt

# Verify installations
echo ""
echo "Verifying installations..."
python3 -c "import torch; print(f'PyTorch: {torch.__version__}')"
python3 -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python3 -c "import weightwatcher as ww; print(f'WeightWatcher: {ww.__version__}')"

# Check for HuggingFace token
echo ""
echo "=================================="
echo "Setup Complete!"
echo "=================================="
echo ""
echo "Next steps:"
echo "1. Activate the virtual environment:"
echo "   source venv/bin/activate"
echo ""
echo "2. (Optional) Login to HuggingFace for Llama access:"
echo "   huggingface-cli login"
echo ""
echo "3. Run the analysis:"
echo "   python scripts/analyze_quantization.py --quantizations fp16 8bit"
echo ""
echo "4. Generate visualizations:"
echo "   python scripts/visualize_results.py"
echo ""
echo "5. Or use the Jupyter notebook:"
echo "   jupyter notebook notebooks/quantization_analysis.ipynb"
echo ""
