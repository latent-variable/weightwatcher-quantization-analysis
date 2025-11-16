#!/bin/bash
# Quick run script for WeightWatcher quantization analysis

set -e

echo "=================================="
echo "WeightWatcher Quantization Analysis"
echo "=================================="
echo ""

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
else
    echo "Warning: Virtual environment not found. Run ./setup.sh first."
    exit 1
fi

# Default parameters
MODEL="Qwen/Qwen2.5-0.5B-Instruct"
QUANTIZATIONS="fp16 8bit"
OUTPUT_DIR="results/metrics"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --quantizations)
            shift
            QUANTIZATIONS=""
            while [[ $# -gt 0 ]] && [[ ! "$1" =~ ^-- ]]; do
                QUANTIZATIONS="$QUANTIZATIONS $1"
                shift
            done
            ;;
        --help)
            echo "Usage: ./run_analysis.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --model MODEL              Model to analyze (default: Qwen/Qwen2.5-0.5B-Instruct)"
            echo "  --quantizations Q1 Q2...   Quantization levels (default: fp16 8bit)"
            echo "                             Available: fp16, 8bit, 6bit, 5bit, 4bit, 2bit"
            echo "  --help                     Show this help message"
            echo ""
            echo "Examples:"
            echo "  ./run_analysis.sh"
            echo "  ./run_analysis.sh --quantizations fp16 8bit 6bit 4bit"
            echo "  ./run_analysis.sh --quantizations fp16 6bit 5bit 4bit 2bit"
            echo "  ./run_analysis.sh --model Qwen/Qwen2.5-1.5B-Instruct"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Create output directories
mkdir -p results/metrics
mkdir -p results/plots

# Run analysis
echo "Running analysis..."
echo "  Model: $MODEL"
echo "  Quantizations: $QUANTIZATIONS"
echo ""

cd scripts
python analyze_quantization.py \
    --model "$MODEL" \
    --quantizations $QUANTIZATIONS \
    --output-dir "../$OUTPUT_DIR"

# Check if analysis was successful
if [ $? -eq 0 ]; then
    echo ""
    echo "=================================="
    echo "Analysis complete!"
    echo "=================================="
    echo ""
    echo "Generating visualizations..."

    python visualize_results.py \
        --results-dir "../$OUTPUT_DIR" \
        --output-dir "../results/plots"

    if [ $? -eq 0 ]; then
        echo ""
        echo "=================================="
        echo "All done!"
        echo "=================================="
        echo ""
        echo "Results saved to:"
        echo "  - Metrics: results/metrics/"
        echo "  - Plots: results/plots/"
        echo "  - Report: results/analysis_report.txt"
        echo ""
        echo "View the analysis report:"
        echo "  cat results/analysis_report.txt"
        echo ""
    else
        echo "Visualization failed. Check error messages above."
        exit 1
    fi
else
    echo "Analysis failed. Check error messages above."
    exit 1
fi
