"""
Distribution metrics for evaluating generated sequences against real data.

This module provides comprehensive metrics and visualization tools for analyzing
the quality of generated sequences compared to real training data.
"""

import numpy as np
import torch
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


class DistributionAnalyzer:
    """Analyzer for comparing real and generated distributions."""

    def __init__(self):
        """Initialize the distribution analyzer."""
        self.metrics_history = []

    def load_sequences(self, file_path: str) -> np.ndarray:
        """
        Load sequences from file.

        Args:
            file_path: Path to sequence file

        Returns:
            Array of sequences [num_sequences, seq_len]
        """
        sequences = []
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip().split()
                if len(line) > 0:
                    sequences.append([float(x) for x in line])
        return np.array(sequences)

    def compute_basic_statistics(self, sequences: np.ndarray) -> dict[str, float]:
        """
        Compute basic statistics for sequences.

        Args:
            sequences: Array of sequences [num_sequences, seq_len]

        Returns:
            Dictionary of statistics
        """
        flat = sequences.flatten()

        return {
            'mean': float(np.mean(flat)),
            'std': float(np.std(flat)),
            'min': float(np.min(flat)),
            'max': float(np.max(flat)),
            'median': float(np.median(flat)),
            'q25': float(np.percentile(flat, 25)),
            'q75': float(np.percentile(flat, 75)),
            'skewness': float(stats.skew(flat)),
            'kurtosis': float(stats.kurtosis(flat))
        }

    def compute_distribution_distance(
        self,
        real_sequences: np.ndarray,
        generated_sequences: np.ndarray
    ) -> dict[str, float]:
        """
        Compute various distance metrics between distributions.

        Args:
            real_sequences: Real data sequences
            generated_sequences: Generated sequences

        Returns:
            Dictionary of distance metrics
        """
        real_flat = real_sequences.flatten()
        gen_flat = generated_sequences.flatten()

        metrics = {}

        # Kolmogorov-Smirnov test
        ks_statistic, ks_pvalue = stats.ks_2samp(real_flat, gen_flat)
        metrics['ks_statistic'] = float(ks_statistic)
        metrics['ks_pvalue'] = float(ks_pvalue)

        # Wasserstein distance (Earth Mover's Distance)
        metrics['wasserstein_distance'] = float(
            stats.wasserstein_distance(real_flat, gen_flat)
        )

        # Jensen-Shannon divergence
        # Create histograms with same bins
        bins = np.linspace(
            min(real_flat.min(), gen_flat.min()),
            max(real_flat.max(), gen_flat.max()),
            50
        )
        real_hist, _ = np.histogram(real_flat, bins=bins, density=True)
        gen_hist, _ = np.histogram(gen_flat, bins=bins, density=True)

        # Normalize to create probability distributions
        real_hist = real_hist / (real_hist.sum() + 1e-10)
        gen_hist = gen_hist / (gen_hist.sum() + 1e-10)

        # Add small epsilon to avoid log(0)
        real_hist = real_hist + 1e-10
        gen_hist = gen_hist + 1e-10

        # Jensen-Shannon divergence
        m = 0.5 * (real_hist + gen_hist)
        js_div = 0.5 * stats.entropy(real_hist, m) + 0.5 * stats.entropy(gen_hist, m)
        metrics['js_divergence'] = float(js_div)

        # Mean Absolute Error between means
        metrics['mae_means'] = float(
            np.abs(np.mean(real_flat) - np.mean(gen_flat))
        )

        # Mean Absolute Error between stds
        metrics['mae_stds'] = float(
            np.abs(np.std(real_flat) - np.std(gen_flat))
        )

        return metrics

    def compute_temporal_statistics(self, sequences: np.ndarray) -> dict[str, float]:
        """
        Compute temporal statistics (time-series properties).

        Args:
            sequences: Array of sequences [num_sequences, seq_len]

        Returns:
            Dictionary of temporal statistics
        """
        # Autocorrelation at lag 1
        autocorr = []
        for seq in sequences:
            if len(seq) > 1:
                ac = np.corrcoef(seq[:-1], seq[1:])[0, 1]
                if not np.isnan(ac):
                    autocorr.append(ac)

        # Mean absolute difference (volatility)
        mad = np.mean(np.abs(np.diff(sequences, axis=1)))

        # Trend (mean of slopes)
        trends = []
        for seq in sequences:
            if len(seq) > 1:
                x = np.arange(len(seq))
                slope, _ = np.polyfit(x, seq, 1)
                trends.append(slope)

        return {
            'autocorr_lag1': float(np.mean(autocorr)) if autocorr else 0.0,
            'volatility': float(mad),
            'mean_trend': float(np.mean(trends)) if trends else 0.0,
            'std_trend': float(np.std(trends)) if trends else 0.0
        }

    def compute_sequence_diversity(self, sequences: np.ndarray) -> dict[str, float]:
        """
        Compute diversity metrics for generated sequences.

        Args:
            sequences: Array of sequences [num_sequences, seq_len]

        Returns:
            Dictionary of diversity metrics
        """
        # Pairwise L2 distances between sequences
        n_samples = min(100, len(sequences))  # Sample for efficiency
        sample_idx = np.random.choice(len(sequences), n_samples, replace=False)
        sample_seqs = sequences[sample_idx]

        pairwise_dists = []
        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                dist = np.linalg.norm(sample_seqs[i] - sample_seqs[j])
                pairwise_dists.append(dist)

        return {
            'mean_pairwise_distance': float(np.mean(pairwise_dists)),
            'std_pairwise_distance': float(np.std(pairwise_dists)),
            'min_pairwise_distance': float(np.min(pairwise_dists)),
            'max_pairwise_distance': float(np.max(pairwise_dists))
        }

    def analyze(
        self,
        real_file: str,
        generated_file: str
    ) -> dict[str, dict[str, float]]:
        """
        Comprehensive analysis of real vs generated distributions.

        Args:
            real_file: Path to real sequences file
            generated_file: Path to generated sequences file

        Returns:
            Dictionary containing all metrics
        """
        # Load sequences
        real_seq = self.load_sequences(real_file)
        gen_seq = self.load_sequences(generated_file)

        print(f"Loaded {len(real_seq)} real sequences")
        print(f"Loaded {len(gen_seq)} generated sequences")

        # Compute all metrics
        results = {
            'real_statistics': self.compute_basic_statistics(real_seq),
            'generated_statistics': self.compute_basic_statistics(gen_seq),
            'distance_metrics': self.compute_distribution_distance(real_seq, gen_seq),
            'real_temporal': self.compute_temporal_statistics(real_seq),
            'generated_temporal': self.compute_temporal_statistics(gen_seq),
            'generated_diversity': self.compute_sequence_diversity(gen_seq)
        }

        # Store in history
        self.metrics_history.append(results)

        return results

    def print_analysis(self, results: dict[str, dict[str, float]]):
        """
        Pretty print analysis results.

        Args:
            results: Results dictionary from analyze()
        """
        print("\n" + "="*70)
        print("DISTRIBUTION ANALYSIS REPORT")
        print("="*70)

        print("\n📊 BASIC STATISTICS")
        print("-" * 70)
        print(f"{'Metric':<20} {'Real':>15} {'Generated':>15} {'Difference':>15}")
        print("-" * 70)

        for key in ['mean', 'std', 'min', 'max', 'median', 'skewness', 'kurtosis']:
            real_val = results['real_statistics'][key]
            gen_val = results['generated_statistics'][key]
            diff = gen_val - real_val
            print(f"{key:<20} {real_val:>15.4f} {gen_val:>15.4f} {diff:>15.4f}")

        print("\n📏 DISTRIBUTION DISTANCES")
        print("-" * 70)
        dist_metrics = results['distance_metrics']
        print(f"Kolmogorov-Smirnov:    {dist_metrics['ks_statistic']:.6f} (p={dist_metrics['ks_pvalue']:.4f})")
        print(f"Wasserstein Distance:  {dist_metrics['wasserstein_distance']:.6f}")
        print(f"Jensen-Shannon Div:    {dist_metrics['js_divergence']:.6f}")
        print(f"MAE (means):           {dist_metrics['mae_means']:.6f}")
        print(f"MAE (stds):            {dist_metrics['mae_stds']:.6f}")

        print("\n⏱️  TEMPORAL PROPERTIES")
        print("-" * 70)
        print(f"{'Metric':<20} {'Real':>15} {'Generated':>15}")
        print("-" * 70)

        for key in ['autocorr_lag1', 'volatility', 'mean_trend']:
            real_val = results['real_temporal'][key]
            gen_val = results['generated_temporal'][key]
            print(f"{key:<20} {real_val:>15.4f} {gen_val:>15.4f}")

        print("\n🎨 GENERATION DIVERSITY")
        print("-" * 70)
        div_metrics = results['generated_diversity']
        print(f"Mean pairwise distance: {div_metrics['mean_pairwise_distance']:.4f}")
        print(f"Std pairwise distance:  {div_metrics['std_pairwise_distance']:.4f}")
        print(f"Min pairwise distance:  {div_metrics['min_pairwise_distance']:.4f}")
        print(f"Max pairwise distance:  {div_metrics['max_pairwise_distance']:.4f}")

        print("\n✅ QUALITY ASSESSMENT")
        print("-" * 70)

        # Simple quality indicators
        ks_good = dist_metrics['ks_statistic'] < 0.1
        js_good = dist_metrics['js_divergence'] < 0.1
        wasserstein_good = dist_metrics['wasserstein_distance'] < 1.0

        print(f"KS Test:              {'✓ GOOD' if ks_good else '✗ NEEDS IMPROVEMENT'}")
        print(f"JS Divergence:        {'✓ GOOD' if js_good else '✗ NEEDS IMPROVEMENT'}")
        print(f"Wasserstein Distance: {'✓ GOOD' if wasserstein_good else '✗ NEEDS IMPROVEMENT'}")

        print("\n" + "="*70)
