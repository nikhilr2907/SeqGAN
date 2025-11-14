#!/usr/bin/env python
"""
Distribution Analysis Tool for SeqGAN.

This script provides comprehensive analysis and visualization of generated
sequences compared to real training data.

Usage:
    python analyze_distributions.py --real data/real_data.txt --generated save/generated_samples.txt
    python analyze_distributions.py --real data/real_data.txt --generated save/generated_samples.txt --output-dir ./analysis_results
"""

import argparse
import os
import sys
from src.utils.distribution_metrics import DistributionAnalyzer
from src.utils.visualise_distributions import DistributionVisualizer


def main():
    """Main analysis function."""
    parser = argparse.ArgumentParser(
        description='Analyze distribution quality of generated sequences',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic analysis
  python analyze_distributions.py --real data/real_data.txt --generated save/generated_samples.txt

  # With custom output directory
  python analyze_distributions.py --real data/real_data.txt --generated save/generated_samples.txt --output-dir ./my_analysis

  # Skip visualization (metrics only)
  python analyze_distributions.py --real data/real_data.txt --generated save/generated_samples.txt --no-plots
        """
    )

    parser.add_argument(
        '--real',
        type=str,
        required=True,
        help='Path to real sequences file'
    )

    parser.add_argument(
        '--generated',
        type=str,
        required=True,
        help='Path to generated sequences file'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='./analysis_results',
        help='Directory to save analysis results (default: ./analysis_results)'
    )

    parser.add_argument(
        '--no-plots',
        action='store_true',
        help='Skip generating plots (compute metrics only)'
    )

    parser.add_argument(
        '--prefix',
        type=str,
        default='analysis',
        help='Prefix for output files (default: analysis)'
    )

    args = parser.parse_args()

    # Validate input files
    if not os.path.exists(args.real):
        print(f"Error: Real data file not found: {args.real}")
        sys.exit(1)

    if not os.path.exists(args.generated):
        print(f"Error: Generated data file not found: {args.generated}")
        sys.exit(1)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print("\n" + "="*70)
    print("SeqGAN Distribution Analysis")
    print("="*70)
    print(f"\nReal data:       {args.real}")
    print(f"Generated data:  {args.generated}")
    print(f"Output dir:      {args.output_dir}")
    print()

    # ==================== METRICS ANALYSIS ====================
    print("\n[1/2] Computing distribution metrics...")
    print("-"*70)

    analyzer = DistributionAnalyzer()
    results = analyzer.analyze(args.real, args.generated)
    analyzer.print_analysis(results)

    # Save metrics to file
    metrics_file = os.path.join(args.output_dir, f'{args.prefix}_metrics.txt')
    with open(metrics_file, 'w') as f:
        f.write("="*70 + "\n")
        f.write("DISTRIBUTION ANALYSIS METRICS\n")
        f.write("="*70 + "\n\n")

        f.write("BASIC STATISTICS\n")
        f.write("-"*70 + "\n")
        f.write(f"{'Metric':<20} {'Real':>15} {'Generated':>15} {'Difference':>15}\n")
        f.write("-"*70 + "\n")

        for key in ['mean', 'std', 'min', 'max', 'median', 'skewness', 'kurtosis']:
            real_val = results['real_statistics'][key]
            gen_val = results['generated_statistics'][key]
            diff = gen_val - real_val
            f.write(f"{key:<20} {real_val:>15.4f} {gen_val:>15.4f} {diff:>15.4f}\n")

        f.write("\n\nDISTRIBUTION DISTANCES\n")
        f.write("-"*70 + "\n")
        dist_metrics = results['distance_metrics']
        f.write(f"Kolmogorov-Smirnov:    {dist_metrics['ks_statistic']:.6f} (p={dist_metrics['ks_pvalue']:.4f})\n")
        f.write(f"Wasserstein Distance:  {dist_metrics['wasserstein_distance']:.6f}\n")
        f.write(f"Jensen-Shannon Div:    {dist_metrics['js_divergence']:.6f}\n")
        f.write(f"MAE (means):           {dist_metrics['mae_means']:.6f}\n")
        f.write(f"MAE (stds):            {dist_metrics['mae_stds']:.6f}\n")

        f.write("\n\nTEMPORAL PROPERTIES\n")
        f.write("-"*70 + "\n")
        f.write(f"{'Metric':<20} {'Real':>15} {'Generated':>15}\n")
        f.write("-"*70 + "\n")

        for key in ['autocorr_lag1', 'volatility', 'mean_trend']:
            real_val = results['real_temporal'][key]
            gen_val = results['generated_temporal'][key]
            f.write(f"{key:<20} {real_val:>15.4f} {gen_val:>15.4f}\n")

        f.write("\n\nGENERATION DIVERSITY\n")
        f.write("-"*70 + "\n")
        div_metrics = results['generated_diversity']
        f.write(f"Mean pairwise distance: {div_metrics['mean_pairwise_distance']:.4f}\n")
        f.write(f"Std pairwise distance:  {div_metrics['std_pairwise_distance']:.4f}\n")

    print(f"\nMetrics saved to: {metrics_file}")

    # ==================== VISUALIZATION ====================
    if not args.no_plots:
        print("\n[2/2] Generating visualizations...")
        print("-"*70)

        visualizer = DistributionVisualizer(save_dir=args.output_dir)
        visualizer.create_full_report(
            args.real,
            args.generated,
            output_prefix=args.prefix
        )

        print(f"\nPlots saved to: {args.output_dir}/")
    else:
        print("\n[2/2] Skipping visualizations (--no-plots specified)")

    # ==================== SUMMARY ====================
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print(f"\nResults saved to: {args.output_dir}/")
    print(f"  - Metrics:         {args.prefix}_metrics.txt")
    if not args.no_plots:
        print(f"  - Distributions:   {args.prefix}_distributions.png")
        print(f"  - Temporal:        {args.prefix}_temporal.png")
    print("\n" + "="*70)


if __name__ == '__main__':
    main()
