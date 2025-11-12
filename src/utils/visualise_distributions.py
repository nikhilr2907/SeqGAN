

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


class DistributionVisualizer:
    """Visualizer for distribution comparisons."""

    def __init__(self, save_dir: str = './plots'):
        """
        Initialize the visualizer.

        Args:
            save_dir: Directory to save plots
        """
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def load_sequences(self, file_path: str) -> np.ndarray:
        """Load sequences from file."""
        sequences = []
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip().split()
                if len(line) > 0:
                    sequences.append([float(x) for x in line])
        return np.array(sequences)

    def plot_distribution_comparison(
        self,
        real_seq: np.ndarray,
        gen_seq: np.ndarray,
        save_name: str = 'distribution_comparison.png'
    ):
        """
        Plot comprehensive distribution comparison.

        Args:
            real_seq: Real sequences
            gen_seq: Generated sequences
            save_name: Filename to save plot
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Distribution Comparison: Real vs Generated', fontsize=16, fontweight='bold')

        real_flat = real_seq.flatten()
        gen_flat = gen_seq.flatten()

        # 1. Histograms
        ax = axes[0, 0]
        ax.hist(real_flat, bins=50, alpha=0.6, label='Real', density=True, color='blue', edgecolor='black')
        ax.hist(gen_flat, bins=50, alpha=0.6, label='Generated', density=True, color='red', edgecolor='black')
        ax.set_xlabel('Value')
        ax.set_ylabel('Density')
        ax.set_title('Value Distribution (Histogram)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 2. KDE plots
        ax = axes[0, 1]
        try:
            real_kde = stats.gaussian_kde(real_flat)
            gen_kde = stats.gaussian_kde(gen_flat)
            x_range = np.linspace(
                min(real_flat.min(), gen_flat.min()),
                max(real_flat.max(), gen_flat.max()),
                200
            )
            ax.plot(x_range, real_kde(x_range), label='Real', color='blue', linewidth=2)
            ax.plot(x_range, gen_kde(x_range), label='Generated', color='red', linewidth=2)
            ax.fill_between(x_range, real_kde(x_range), alpha=0.3, color='blue')
            ax.fill_between(x_range, gen_kde(x_range), alpha=0.3, color='red')
        except:
            ax.text(0.5, 0.5, 'KDE estimation failed', ha='center', va='center', transform=ax.transAxes)
        ax.set_xlabel('Value')
        ax.set_ylabel('Density')
        ax.set_title('Kernel Density Estimation')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 3. Q-Q plot
        ax = axes[0, 2]
        stats.probplot(real_flat, dist="norm", plot=None)
        stats.probplot(gen_flat, dist="norm", plot=None)
        real_quantiles = np.sort(real_flat)
        gen_quantiles = np.sort(gen_flat)
        # Match lengths
        n = min(len(real_quantiles), len(gen_quantiles))
        ax.scatter(real_quantiles[:n], gen_quantiles[:n], alpha=0.5, s=10)
        lims = [
            np.min([ax.get_xlim(), ax.get_ylim()]),
            np.max([ax.get_xlim(), ax.get_ylim()])
        ]
        ax.plot(lims, lims, 'r--', alpha=0.75, zorder=0, linewidth=2)
        ax.set_xlabel('Real Quantiles')
        ax.set_ylabel('Generated Quantiles')
        ax.set_title('Q-Q Plot')
        ax.grid(True, alpha=0.3)

        # 4. Box plots
        ax = axes[1, 0]
        data_to_plot = [real_flat, gen_flat]
        bp = ax.boxplot(data_to_plot, labels=['Real', 'Generated'], patch_artist=True)
        bp['boxes'][0].set_facecolor('lightblue')
        bp['boxes'][1].set_facecolor('lightcoral')
        ax.set_ylabel('Value')
        ax.set_title('Box Plot Comparison')
        ax.grid(True, alpha=0.3, axis='y')

        # 5. Cumulative Distribution
        ax = axes[1, 1]
        real_sorted = np.sort(real_flat)
        gen_sorted = np.sort(gen_flat)
        real_cdf = np.arange(1, len(real_sorted) + 1) / len(real_sorted)
        gen_cdf = np.arange(1, len(gen_sorted) + 1) / len(gen_sorted)
        ax.plot(real_sorted, real_cdf, label='Real', color='blue', linewidth=2)
        ax.plot(gen_sorted, gen_cdf, label='Generated', color='red', linewidth=2)
        ax.set_xlabel('Value')
        ax.set_ylabel('Cumulative Probability')
        ax.set_title('Cumulative Distribution Function')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 6. Statistics comparison
        ax = axes[1, 2]
        metrics = ['Mean', 'Std', 'Min', 'Max', 'Median']
        real_stats = [
            np.mean(real_flat),
            np.std(real_flat),
            np.min(real_flat),
            np.max(real_flat),
            np.median(real_flat)
        ]
        gen_stats = [
            np.mean(gen_flat),
            np.std(gen_flat),
            np.min(gen_flat),
            np.max(gen_flat),
            np.median(gen_flat)
        ]

        x = np.arange(len(metrics))
        width = 0.35
        ax.bar(x - width/2, real_stats, width, label='Real', color='blue', alpha=0.7)
        ax.bar(x + width/2, gen_stats, width, label='Generated', color='red', alpha=0.7)
        ax.set_ylabel('Value')
        ax.set_title('Statistical Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        save_path = os.path.join(self.save_dir, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved distribution comparison to {save_path}")
        plt.close()

    def plot_temporal_analysis(
        self,
        real_seq: np.ndarray,
        gen_seq: np.ndarray,
        save_name: str = 'temporal_analysis.png'
    ):
        """
        Plot temporal properties of sequences.

        Args:
            real_seq: Real sequences
            gen_seq: Generated sequences
            save_name: Filename to save plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Temporal Analysis: Real vs Generated', fontsize=16, fontweight='bold')

        # 1. Sample sequences
        ax = axes[0, 0]
        n_samples = min(5, len(real_seq), len(gen_seq))
        for i in range(n_samples):
            ax.plot(real_seq[i], alpha=0.6, color='blue', linewidth=1)
        for i in range(n_samples):
            ax.plot(gen_seq[i], alpha=0.6, color='red', linewidth=1, linestyle='--')
        ax.plot([], [], color='blue', label='Real', linewidth=2)
        ax.plot([], [], color='red', label='Generated', linestyle='--', linewidth=2)
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Value')
        ax.set_title(f'Sample Sequences (n={n_samples})')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 2. Mean sequence with std
        ax = axes[0, 1]
        real_mean = np.mean(real_seq, axis=0)
        real_std = np.std(real_seq, axis=0)
        gen_mean = np.mean(gen_seq, axis=0)
        gen_std = np.std(gen_seq, axis=0)

        time_steps = np.arange(len(real_mean))
        ax.plot(time_steps, real_mean, color='blue', label='Real Mean', linewidth=2)
        ax.fill_between(time_steps, real_mean - real_std, real_mean + real_std,
                        alpha=0.3, color='blue', label='Real ±1 Std')
        ax.plot(time_steps, gen_mean, color='red', label='Generated Mean', linewidth=2)
        ax.fill_between(time_steps, gen_mean - gen_std, gen_mean + gen_std,
                        alpha=0.3, color='red', label='Generated ±1 Std')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Value')
        ax.set_title('Mean Sequences with Standard Deviation')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 3. Autocorrelation
        ax = axes[1, 0]
        max_lag = min(20, real_seq.shape[1] - 1)

        def compute_autocorr(sequences, max_lag):
            autocorrs = []
            for lag in range(1, max_lag + 1):
                corrs = []
                for seq in sequences:
                    if len(seq) > lag:
                        c = np.corrcoef(seq[:-lag], seq[lag:])[0, 1]
                        if not np.isnan(c):
                            corrs.append(c)
                autocorrs.append(np.mean(corrs) if corrs else 0)
            return autocorrs

        real_autocorr = compute_autocorr(real_seq, max_lag)
        gen_autocorr = compute_autocorr(gen_seq, max_lag)

        lags = np.arange(1, max_lag + 1)
        ax.plot(lags, real_autocorr, 'o-', color='blue', label='Real', linewidth=2, markersize=5)
        ax.plot(lags, gen_autocorr, 's-', color='red', label='Generated', linewidth=2, markersize=5)
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax.set_xlabel('Lag')
        ax.set_ylabel('Autocorrelation')
        ax.set_title('Autocorrelation Function')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 4. Value distribution over time
        ax = axes[1, 1]
        seq_len = real_seq.shape[1]
        positions = np.arange(seq_len)

        # Create violin plot data
        real_data = [real_seq[:, i] for i in range(seq_len)]
        gen_data = [gen_seq[:, i] for i in range(seq_len)]

        # Sample positions if too many
        if seq_len > 10:
            step = seq_len // 10
            positions_sample = positions[::step]
            real_data_sample = [real_data[i] for i in range(0, seq_len, step)]
            gen_data_sample = [gen_data[i] for i in range(0, seq_len, step)]
        else:
            positions_sample = positions
            real_data_sample = real_data
            gen_data_sample = gen_data

        vp1 = ax.violinplot(real_data_sample, positions=positions_sample - 0.2, widths=0.3,
                            showmeans=True, showmedians=False)
        vp2 = ax.violinplot(gen_data_sample, positions=positions_sample + 0.2, widths=0.3,
                            showmeans=True, showmedians=False)

        for pc in vp1['bodies']:
            pc.set_facecolor('blue')
            pc.set_alpha(0.6)
        for pc in vp2['bodies']:
            pc.set_facecolor('red')
            pc.set_alpha(0.6)

        ax.plot([], [], color='blue', label='Real', linewidth=5)
        ax.plot([], [], color='red', label='Generated', linewidth=5)
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Value Distribution')
        ax.set_title('Distribution Evolution Over Time')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = os.path.join(self.save_dir, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved temporal analysis to {save_path}")
        plt.close()

    def plot_training_metrics(
        self,
        metrics_history: list,
        save_name: str = 'training_metrics.png'
    ):
        """
        Plot metrics evolution during training.

        Args:
            metrics_history: List of metric dictionaries over time
            save_name: Filename to save plot
        """
        if not metrics_history:
            print("No metrics history to plot")
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training Metrics Evolution', fontsize=16, fontweight='bold')

        epochs = np.arange(len(metrics_history))

        # 1. Distribution distances
        ax = axes[0, 0]
        ks_stats = [m['distance_metrics']['ks_statistic'] for m in metrics_history]
        wasserstein = [m['distance_metrics']['wasserstein_distance'] for m in metrics_history]

        ax.plot(epochs, ks_stats, 'o-', label='KS Statistic', linewidth=2, markersize=5)
        ax.set_xlabel('Checkpoint')
        ax.set_ylabel('KS Statistic')
        ax.set_title('Kolmogorov-Smirnov Statistic (Lower is Better)')
        ax.grid(True, alpha=0.3)

        # 2. Wasserstein distance
        ax = axes[0, 1]
        ax.plot(epochs, wasserstein, 's-', color='orange', label='Wasserstein', linewidth=2, markersize=5)
        ax.set_xlabel('Checkpoint')
        ax.set_ylabel('Wasserstein Distance')
        ax.set_title('Wasserstein Distance (Lower is Better)')
        ax.grid(True, alpha=0.3)

        # 3. JS divergence
        ax = axes[1, 0]
        js_div = [m['distance_metrics']['js_divergence'] for m in metrics_history]
        ax.plot(epochs, js_div, '^-', color='green', linewidth=2, markersize=5)
        ax.set_xlabel('Checkpoint')
        ax.set_ylabel('JS Divergence')
        ax.set_title('Jensen-Shannon Divergence (Lower is Better)')
        ax.grid(True, alpha=0.3)

        # 4. Mean and Std MAE
        ax = axes[1, 1]
        mae_means = [m['distance_metrics']['mae_means'] for m in metrics_history]
        mae_stds = [m['distance_metrics']['mae_stds'] for m in metrics_history]
        ax.plot(epochs, mae_means, 'o-', label='MAE (Means)', linewidth=2, markersize=5)
        ax.plot(epochs, mae_stds, 's-', label='MAE (Stds)', linewidth=2, markersize=5)
        ax.set_xlabel('Checkpoint')
        ax.set_ylabel('Mean Absolute Error')
        ax.set_title('Statistical Moment Differences')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = os.path.join(self.save_dir, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved training metrics to {save_path}")
        plt.close()

    def create_full_report(
        self,
        real_file: str,
        generated_file: str,
        output_prefix: str = 'analysis'
    ):
        """
        Create complete analysis report with all plots.

        Args:
            real_file: Path to real sequences file
            generated_file: Path to generated sequences file
            output_prefix: Prefix for output files
        """
        print("\nGenerating comprehensive visualization report...")
        print("="*70)

        # Load data
        real_seq = self.load_sequences(real_file)
        gen_seq = self.load_sequences(generated_file)

        print(f"Loaded {len(real_seq)} real sequences")
        print(f"Loaded {len(gen_seq)} generated sequences")

        # Generate plots
        self.plot_distribution_comparison(
            real_seq, gen_seq,
            save_name=f'{output_prefix}_distributions.png'
        )

        self.plot_temporal_analysis(
            real_seq, gen_seq,
            save_name=f'{output_prefix}_temporal.png'
        )

        print("\n" + "="*70)
        print(f"All plots saved to {self.save_dir}/")
        print("="*70)
