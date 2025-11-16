#!/usr/bin/env python3
"""
Visualization script for WeightWatcher quantization analysis results.

Generates comprehensive plots comparing alpha metrics across different
quantization levels.

Usage:
    python visualize_results.py --results-dir ../results/metrics
"""

import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 10


def load_results(results_dir: Path) -> Dict[str, pd.DataFrame]:
    """Load all quantization results from CSV files."""
    results = {}

    for csv_file in results_dir.glob("results_*.csv"):
        quant = csv_file.stem.replace("results_", "")
        df = pd.read_csv(csv_file)
        results[quant] = df
        print(f"Loaded {quant}: {len(df)} layers")

    return results


def plot_alpha_distributions(all_results: Dict[str, pd.DataFrame], output_dir: Path):
    """Plot alpha distributions for each quantization level."""
    n_quants = len(all_results)

    # Dynamic grid size based on number of quantizations
    if n_quants <= 4:
        nrows, ncols = 2, 2
    elif n_quants <= 6:
        nrows, ncols = 2, 3
    else:
        nrows, ncols = 3, 3

    fig, axes = plt.subplots(nrows, ncols, figsize=(8*ncols, 6*nrows))
    axes = axes.flatten()

    quantizations = list(all_results.keys())
    colors = sns.color_palette("husl", n_quants)

    for idx, (quant, results) in enumerate(all_results.items()):
        ax = axes[idx]

        # Plot histogram
        ax.hist(results['alpha'], bins=30, alpha=0.7, color=colors[idx],
                edgecolor='black', linewidth=0.5)

        # Mark optimal range
        ax.axvspan(2, 6, alpha=0.2, color='green', label='Optimal Range [2,6]')
        ax.axvline(results['alpha'].mean(), color='red', linestyle='--',
                   linewidth=2, label=f'Mean: {results["alpha"].mean():.2f}')

        # Labels
        ax.set_title(f'Alpha Distribution - {quant.upper()}', fontsize=14, fontweight='bold')
        ax.set_xlabel('Alpha (α)', fontsize=12)
        ax.set_ylabel('Number of Layers', fontsize=12)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

        # Statistics text
        stats_text = f'Mean: {results["alpha"].mean():.2f}\n'
        stats_text += f'Std: {results["alpha"].std():.2f}\n'
        stats_text += f'Median: {results["alpha"].median():.2f}\n'
        stats_text += f'Range: [{results["alpha"].min():.2f}, {results["alpha"].max():.2f}]'

        ax.text(0.98, 0.97, stats_text,
                transform=ax.transAxes,
                verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                fontsize=9)

    # Hide unused subplots
    for idx in range(n_quants, nrows * ncols):
        axes[idx].axis('off')

    plt.tight_layout()
    output_file = output_dir / "alpha_distributions.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def plot_alpha_comparison_boxplot(all_results: Dict[str, pd.DataFrame], output_dir: Path):
    """Create boxplot comparing alpha across quantizations."""
    fig, ax = plt.subplots(figsize=(12, 8))

    # Prepare data
    data = []
    labels = []

    for quant, results in all_results.items():
        data.append(results['alpha'])
        labels.append(quant.upper())

    # Create boxplot
    bp = ax.boxplot(data, labels=labels, patch_artist=True,
                     notch=True, showmeans=True)

    # Color boxes
    colors = sns.color_palette("husl", len(data))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Optimal range
    ax.axhspan(2, 6, alpha=0.15, color='green', label='Optimal Range [2,6]')

    # Labels
    ax.set_title('Alpha Distribution Comparison Across Quantizations',
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_ylabel('Alpha (α)', fontsize=14)
    ax.set_xlabel('Quantization Level', fontsize=14)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    output_file = output_dir / "alpha_comparison_boxplot.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def plot_layer_wise_alpha(all_results: Dict[str, pd.DataFrame], output_dir: Path):
    """Plot alpha values layer-by-layer for each quantization."""
    fig, ax = plt.subplots(figsize=(16, 8))

    colors = sns.color_palette("husl", len(all_results))

    for idx, (quant, results) in enumerate(all_results.items()):
        # Sort by layer_id if available
        if 'layer_id' in results.columns:
            plot_data = results.sort_values('layer_id')
            x = plot_data['layer_id']
        else:
            x = range(len(results))
            plot_data = results

        ax.plot(x, plot_data['alpha'], marker='o', markersize=3,
                linewidth=1.5, alpha=0.7, label=quant.upper(), color=colors[idx])

    # Optimal range
    ax.axhspan(2, 6, alpha=0.1, color='green', label='Optimal Range')

    # Labels
    ax.set_title('Layer-wise Alpha Comparison Across Quantizations',
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Layer Index', fontsize=14)
    ax.set_ylabel('Alpha (α)', fontsize=14)
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_file = output_dir / "layerwise_alpha_comparison.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def plot_alpha_statistics_summary(all_results: Dict[str, pd.DataFrame], output_dir: Path):
    """Create summary statistics visualization."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    quantizations = list(all_results.keys())
    colors = sns.color_palette("husl", len(quantizations))

    # 1. Mean Alpha
    ax = axes[0, 0]
    means = [results['alpha'].mean() for results in all_results.values()]
    bars = ax.bar(quantizations, means, color=colors, alpha=0.7, edgecolor='black')
    ax.axhline(y=4, color='green', linestyle='--', linewidth=2, alpha=0.5, label='Ideal (~4)')
    ax.set_title('Mean Alpha by Quantization', fontsize=14, fontweight='bold')
    ax.set_ylabel('Mean Alpha', fontsize=12)
    ax.set_xlabel('Quantization', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    # 2. Standard Deviation
    ax = axes[0, 1]
    stds = [results['alpha'].std() for results in all_results.values()]
    bars = ax.bar(quantizations, stds, color=colors, alpha=0.7, edgecolor='black')
    ax.set_title('Alpha Standard Deviation by Quantization', fontsize=14, fontweight='bold')
    ax.set_ylabel('Std Dev', fontsize=12)
    ax.set_xlabel('Quantization', fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')

    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    # 3. Layers in Optimal Range
    ax = axes[1, 0]
    optimal_counts = [
        ((results['alpha'] >= 2) & (results['alpha'] <= 6)).sum()
        for results in all_results.values()
    ]
    total_layers = [len(results) for results in all_results.values()]
    optimal_pcts = [count / total * 100 for count, total in zip(optimal_counts, total_layers)]

    bars = ax.bar(quantizations, optimal_pcts, color=colors, alpha=0.7, edgecolor='black')
    ax.axhline(y=100, color='green', linestyle='--', linewidth=2, alpha=0.5, label='100%')
    ax.set_title('Percentage of Layers in Optimal Range [2,6]', fontsize=14, fontweight='bold')
    ax.set_ylabel('Percentage (%)', fontsize=12)
    ax.set_xlabel('Quantization', fontsize=12)
    ax.set_ylim([0, 105])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    for bar, count, total in zip(bars, optimal_counts, total_layers):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%\n({count}/{total})',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

    # 4. Alpha Range
    ax = axes[1, 1]
    x_pos = np.arange(len(quantizations))
    mins = [results['alpha'].min() for results in all_results.values()]
    maxs = [results['alpha'].max() for results in all_results.values()]
    ranges = [max_val - min_val for min_val, max_val in zip(mins, maxs)]

    # Plot ranges as vertical lines
    for i, (quant, min_val, max_val) in enumerate(zip(quantizations, mins, maxs)):
        ax.plot([i, i], [min_val, max_val], linewidth=3, color=colors[i], alpha=0.7)
        ax.scatter([i], [min_val], s=100, color=colors[i], marker='v', zorder=3)
        ax.scatter([i], [max_val], s=100, color=colors[i], marker='^', zorder=3)

    ax.axhspan(2, 6, alpha=0.15, color='green', label='Optimal Range')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(quantizations)
    ax.set_title('Alpha Range (Min-Max) by Quantization', fontsize=14, fontweight='bold')
    ax.set_ylabel('Alpha', fontsize=12)
    ax.set_xlabel('Quantization', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    output_file = output_dir / "alpha_statistics_summary.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def plot_spectral_norms(all_results: Dict[str, pd.DataFrame], output_dir: Path):
    """Plot spectral norm comparisons if available."""
    # Check if log_norm is available
    has_log_norm = any('log_norm' in results.columns for results in all_results.values())

    if not has_log_norm:
        print("Spectral norm data not available in results")
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # 1. Log Norm Distribution
    ax = axes[0]
    for quant, results in all_results.items():
        if 'log_norm' in results.columns:
            ax.hist(results['log_norm'], bins=20, alpha=0.5, label=quant.upper())

    ax.set_title('Log Spectral Norm Distribution', fontsize=14, fontweight='bold')
    ax.set_xlabel('Log Norm', fontsize=12)
    ax.set_ylabel('Number of Layers', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Mean Log Norm by Quantization
    ax = axes[1]
    quantizations = []
    log_norms = []

    for quant, results in all_results.items():
        if 'log_norm' in results.columns:
            quantizations.append(quant.upper())
            log_norms.append(results['log_norm'].mean())

    colors = sns.color_palette("husl", len(quantizations))
    bars = ax.bar(quantizations, log_norms, color=colors, alpha=0.7, edgecolor='black')

    ax.set_title('Mean Log Spectral Norm by Quantization', fontsize=14, fontweight='bold')
    ax.set_ylabel('Mean Log Norm', fontsize=12)
    ax.set_xlabel('Quantization', fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')

    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    output_file = output_dir / "spectral_norms_comparison.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def generate_report(all_results: Dict[str, pd.DataFrame], output_dir: Path):
    """Generate a text report with key findings."""
    report_lines = []

    report_lines.append("="*80)
    report_lines.append("WEIGHTWATCHER QUANTIZATION ANALYSIS REPORT")
    report_lines.append("="*80)
    report_lines.append("")

    # Summary statistics
    report_lines.append("SUMMARY STATISTICS")
    report_lines.append("-"*80)

    for quant, results in all_results.items():
        report_lines.append(f"\n{quant.upper()}:")
        report_lines.append(f"  Total Layers: {len(results)}")
        report_lines.append(f"  Alpha Mean: {results['alpha'].mean():.3f}")
        report_lines.append(f"  Alpha Std: {results['alpha'].std():.3f}")
        report_lines.append(f"  Alpha Median: {results['alpha'].median():.3f}")
        report_lines.append(f"  Alpha Range: [{results['alpha'].min():.3f}, {results['alpha'].max():.3f}]")

        optimal = ((results['alpha'] >= 2) & (results['alpha'] <= 6)).sum()
        pct = (optimal / len(results)) * 100
        report_lines.append(f"  Layers in Optimal Range [2,6]: {optimal}/{len(results)} ({pct:.1f}%)")

        below_2 = (results['alpha'] < 2).sum()
        above_6 = (results['alpha'] > 6).sum()
        report_lines.append(f"  Layers < 2: {below_2}")
        report_lines.append(f"  Layers > 6: {above_6}")

    # Comparison to baseline
    if 'fp16' in all_results:
        report_lines.append("\n" + "="*80)
        report_lines.append("COMPARISON TO FP16 BASELINE")
        report_lines.append("-"*80)

        baseline_mean = all_results['fp16']['alpha'].mean()

        for quant, results in all_results.items():
            if quant == 'fp16':
                continue

            mean_diff = results['alpha'].mean() - baseline_mean
            pct_change = (mean_diff / baseline_mean) * 100

            report_lines.append(f"\n{quant.upper()}:")
            report_lines.append(f"  Mean Alpha Change: {mean_diff:+.3f} ({pct_change:+.2f}%)")

            # Layer-by-layer correlation
            if len(results) == len(all_results['fp16']):
                correlation = results['alpha'].corr(all_results['fp16']['alpha'])
                report_lines.append(f"  Correlation with FP16: {correlation:.3f}")

    report_lines.append("\n" + "="*80)

    # Save report
    report_file = output_dir / "analysis_report.txt"
    with open(report_file, 'w') as f:
        f.write('\n'.join(report_lines))

    print(f"\nSaved: {report_file}")

    # Also print to console
    print("\n" + '\n'.join(report_lines))


def main():
    parser = argparse.ArgumentParser(
        description="Visualize WeightWatcher quantization analysis results"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="../results/metrics",
        help="Directory containing CSV result files"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="../results/plots",
        help="Directory for output plots"
    )

    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("WEIGHTWATCHER QUANTIZATION VISUALIZATION")
    print("="*80)
    print(f"Loading results from: {results_dir}")
    print(f"Saving plots to: {output_dir}")
    print("="*80)

    # Load results
    all_results = load_results(results_dir)

    if not all_results:
        print("No results found! Please run analyze_quantization.py first.")
        return

    print(f"\nLoaded {len(all_results)} quantization results")

    # Generate visualizations
    print("\nGenerating visualizations...")

    plot_alpha_distributions(all_results, output_dir)
    plot_alpha_comparison_boxplot(all_results, output_dir)
    plot_layer_wise_alpha(all_results, output_dir)
    plot_alpha_statistics_summary(all_results, output_dir)
    plot_spectral_norms(all_results, output_dir)

    # Generate report
    generate_report(all_results, output_dir.parent)

    print("\n" + "="*80)
    print("VISUALIZATION COMPLETE")
    print("="*80)
    print(f"\nAll plots saved to: {output_dir}")
    print("\nGenerated files:")
    print("  - alpha_distributions.png")
    print("  - alpha_comparison_boxplot.png")
    print("  - layerwise_alpha_comparison.png")
    print("  - alpha_statistics_summary.png")
    print("  - spectral_norms_comparison.png (if available)")
    print("  - analysis_report.txt")


if __name__ == "__main__":
    main()
