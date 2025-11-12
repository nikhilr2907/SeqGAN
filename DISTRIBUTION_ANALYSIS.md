# Distribution Analysis Guide

This guide explains how to analyze and visualize the quality of generated sequences using comprehensive distribution metrics and plots.

## Quick Start

```bash
# Basic analysis
python analyze_distributions.py \
    --real data/real_data.txt \
    --generated save/generated_samples.txt

# With custom output directory
python analyze_distributions.py \
    --real data/real_data.txt \
    --generated save/generated_samples.txt \
    --output-dir ./my_analysis
```

## Table of Contents

- [Overview](#overview)
- [Metrics Explained](#metrics-explained)
- [Visualisation Guide](#Visualisation-guide)
- [Interpreting Results](#interpreting-results)
- [Advanced Usage](#advanced-usage)

---

## Overview

The distribution analysis tool provides:

1. **Statistical Metrics**: Compare basic statistics (mean, std, etc.)
2. **Distribution Distances**: Quantify similarity between distributions
3. **Temporal Properties**: Analyze time-series characteristics
4. **Diversity Metrics**: Measure generation variety
5. **Comprehensive Visualisations**: 10+ plots for deep analysis

---

## Metrics Explained

### Basic Statistics

| Metric | Description | Good Performance |
|--------|-------------|------------------|
| **Mean** | Average value across all sequences | Close to real data mean |
| **Std** | Standard deviation | Close to real data std |
| **Min/Max** | Value range | Similar range to real data |
| **Median** | 50th percentile | Close to real data median |
| **Skewness** | Distribution asymmetry | Close to real data skewness |
| **Kurtosis** | Distribution tail heaviness | Close to real data kurtosis |

**Example:**
```
Metric               Real    Generated    Difference
mean                 0.125      0.142        0.017
std                  1.023      1.089        0.066
```

Good: Small differences indicate similar distributions.

---

### Distribution Distance Metrics

#### Kolmogorov-Smirnov (KS) Statistic

**What it measures:** Maximum difference between cumulative distributions

**Range:** [0, 1]

**Interpretation:**
- `< 0.05`: **Excellent** - Distributions are very similar
- `0.05 - 0.10`: **Good** - Minor differences
- `0.10 - 0.20`: **Fair** - Noticeable differences
- `> 0.20`: **Poor** - Significantly different distributions

**P-value:**
- `> 0.05`: Cannot reject hypothesis that distributions are the same (good!)
- `< 0.05`: Distributions are significantly different

**Example:**
```
KS Statistic: 0.0342 (p=0.523)
✓ GOOD - Distributions are statistically similar
```

---

#### Wasserstein Distance (Earth Mover's Distance)

**What it measures:** Minimum "work" to transform one distribution into another

**Range:** [0, ∞)

**Interpretation:**
- `< 0.5`: **Excellent** - Very close distributions
- `0.5 - 1.0`: **Good** - Reasonably close
- `1.0 - 2.0`: **Fair** - Moderate difference
- `> 2.0`: **Poor** - Large difference

**Example:**
```
Wasserstein Distance: 0.234
✓ EXCELLENT - Distributions are very close
```

---

#### Jensen-Shannon (JS) Divergence

**What it measures:** Symmetric divergence between probability distributions

**Range:** [0, 1] (when using log base 2)

**Interpretation:**
- `< 0.05`: **Excellent** - Nearly identical
- `0.05 - 0.10`: **Good** - Similar distributions
- `0.10 - 0.20`: **Fair** - Moderate divergence
- `> 0.20`: **Poor** - High divergence

**Example:**
```
JS Divergence: 0.067
✓ GOOD - Distributions are similar
```

---

### Temporal Properties

#### Autocorrelation (Lag 1)

**What it measures:** Correlation between consecutive time steps

**Range:** [-1, 1]

**Interpretation:**
- Positive: Values tend to follow previous values
- Zero: No temporal correlation
- Negative: Values tend to alternate

**Goal:** Match real data's autocorrelation pattern

**Example:**
```
Metric               Real    Generated
autocorr_lag1       0.234      0.198
```

Good: Similar autocorrelation indicates similar temporal dynamics.

---

#### Volatility (Mean Absolute Difference)

**What it measures:** Average magnitude of changes between time steps

**Range:** [0, ∞)

**Interpretation:**
- Low: Smooth, gradual changes
- High: Rapid, large changes

**Goal:** Match real data's volatility

**Example:**
```
Metric               Real    Generated
volatility          0.456      0.489
```

---

#### Trend

**What it measures:** Average slope across sequences

**Interpretation:**
- Positive: Upward trend
- Zero: No trend
- Negative: Downward trend

**Goal:** Match real data's trend characteristics

---

### Diversity Metrics

#### Mean Pairwise Distance

**What it measures:** Average L2 distance between different generated sequences

**Range:** [0, ∞)

**Interpretation:**
- Too low: Generator produces similar/repetitive sequences (mode collapse)
- Good: Sequences are diverse but realistic
- Too high: Generator produces random noise

**Example:**
```
Mean pairwise distance: 12.45
Min pairwise distance:   3.21
Max pairwise distance:  28.67
```

Good: Wide range indicates healthy diversity without mode collapse.

---

## Visualisation Guide

The analysis generates multiple plots:

### 1. Distribution Comparison (`*_distributions.png`)

**6 subplots:**

1. **Histogram**: Value distribution comparison
   - Overlapping bars show where distributions match/differ
   - Look for similar shapes and peaks

2. **KDE Plot**: Smooth density estimation
   - Clearer view of distribution shape
   - Filled areas show overlap

3. **Q-Q Plot**: Quantile-quantile comparison
   - Points on diagonal = perfect match
   - Deviations show where distributions differ

4. **Box Plot**: Statistical summary
   - Shows median, quartiles, outliers
   - Compare ranges and central tendencies

5. **CDF**: Cumulative distribution function
   - Closer curves = more similar distributions
   - Matches KS test Visualisation

6. **Statistics Bar Chart**: Direct comparison of metrics
   - Visual comparison of mean, std, min, max, median

**How to interpret:**
- **Good**: Overlapping distributions, points near Q-Q diagonal, similar box plots
- **Poor**: Separated distributions, Q-Q points far from diagonal, different ranges

---

### 2. Temporal Analysis (`*_temporal.png`)

**4 subplots:**

1. **Sample Sequences**: Individual sequence examples
   - Blue solid = Real
   - Red dashed = Generated
   - Check if generated sequences look realistic

2. **Mean Sequences with Std**: Average behavior
   - Shows typical sequence shape
   - Shaded regions = variability
   - Look for similar trends and variance

3. **Autocorrelation Function**: Temporal dependencies
   - How values relate to past values
   - Match patterns for realistic dynamics

4. **Distribution Evolution Over Time**: Value distribution at each timestep
   - Violin plots show distribution shape
   - Check if temporal patterns match

**How to interpret:**
- **Good**: Generated sequences follow similar patterns, autocorrelation matches
- **Poor**: Different trends, autocorrelation mismatch, unrealistic patterns

---

## Interpreting Results

### Example: Good Generation

```
📊 BASIC STATISTICS
Metric               Real    Generated    Difference
mean                0.125      0.142        0.017     ✓ Small
std                 1.023      1.089        0.066     ✓ Small

📏 DISTRIBUTION DISTANCES
KS Statistic:       0.0342 (p=0.523)      ✓ GOOD
Wasserstein:        0.234                 ✓ EXCELLENT
JS Divergence:      0.067                 ✓ GOOD

⏱️ TEMPORAL PROPERTIES
autocorr_lag1       0.234      0.198      ✓ Similar
volatility          0.456      0.489      ✓ Similar

✅ QUALITY ASSESSMENT
All metrics indicate good generation quality
```

**Indicators of success:**
- ✅ KS statistic < 0.05
- ✅ Wasserstein < 0.5
- ✅ JS divergence < 0.10
- ✅ Similar temporal properties
- ✅ Overlapping distributions in plots
- ✅ Q-Q points near diagonal

---

### Example: Poor Generation (Mode Collapse)

```
📊 BASIC STATISTICS
Metric               Real    Generated    Difference
mean                0.125      0.125        0.000     ✓ Too perfect
std                 1.023      0.234       -0.789     ✗ Too small

🎨 GENERATION DIVERSITY
Mean pairwise dist:  2.45                 ✗ Too low

Indicators:
- Generated std much smaller than real
- Very low diversity
- Tight distribution in plots
```

**Mode collapse signs:**
- Low standard deviation
- Low pairwise distances
- Narrow distribution in plots

---

### Example: Poor Generation (High Divergence)

```
📏 DISTRIBUTION DISTANCES
KS Statistic:       0.345 (p=0.001)      ✗ POOR
Wasserstein:        2.456                ✗ POOR
JS Divergence:      0.287                ✗ POOR

Indicators:
- Distributions clearly separated in plots
- Q-Q points far from diagonal
- Different ranges/shapes
```

**Divergence signs:**
- High KS statistic
- Separated distributions in plots
- Different statistical properties

---

## Advanced Usage

### Comparing Multiple Checkpoints

```bash
# Analyze different training checkpoints
for epoch in 100 200 300 400 500; do
    python analyze_distributions.py \
        --real data/real_data.txt \
        --generated save/generated_epoch_${epoch}.txt \
        --output-dir analysis/epoch_${epoch} \
        --prefix epoch_${epoch}
done
```

Then compare metrics across epochs to track training progress.

---

### Metrics Only (Fast Analysis)

```bash
# Skip Visualisation for quick metrics
python analyze_distributions.py \
    --real data/real_data.txt \
    --generated save/generated_samples.txt \
    --no-plots
```

Useful for:
- Quick sanity checks
- Automated testing
- High-frequency monitoring

---

### Programmatic Usage

```python
from src.utils.distribution_metrics import DistributionAnalyzer
from src.utils.visualize_distributions import DistributionVisualizer

# Analyze metrics
analyzer = DistributionAnalyzer()
results = analyzer.analyze('real.txt', 'generated.txt')
analyzer.print_analysis(results)

# Get specific metrics
ks_stat = results['distance_metrics']['ks_statistic']
js_div = results['distance_metrics']['js_divergence']

# Generate Visualisations
visualizer = DistributionVisualizer(save_dir='./plots')
visualizer.create_full_report('real.txt', 'generated.txt')
```

---

### Tracking Training Progress

```python
# During training, periodically analyze
from src.utils.distribution_metrics import DistributionAnalyzer

analyzer = DistributionAnalyzer()

for epoch in range(num_epochs):
    # ... training code ...

    if epoch % 10 == 0:
        results = analyzer.analyze('real.txt', f'generated_epoch_{epoch}.txt')

        # Track key metric
        ks_stat = results['distance_metrics']['ks_statistic']
        print(f"Epoch {epoch}: KS = {ks_stat:.4f}")

# Plot metric evolution
analyzer.visualizer.plot_training_metrics(
    analyzer.metrics_history,
    save_name='training_progress.png'
)
```

---

## Troubleshooting

### Issue: "File not found"

**Cause:** Incorrect file paths

**Solution:**
```bash
# Use absolute paths
python analyze_distributions.py \
    --real /full/path/to/real_data.txt \
    --generated /full/path/to/generated.txt
```

---

### Issue: "Empty sequences" or "Invalid data"

**Cause:** File format issues

**Expected format:**
```
1.23 0.45 -0.67 0.89 ...
0.12 0.34 -0.56 0.78 ...
...
```

**Requirements:**
- One sequence per line
- Space-separated values
- All sequences same length

---

### Issue: Metrics show "inf" or "nan"

**Cause:**
- Empty files
- All-zero sequences
- Extremely small variance

**Solution:**
- Check data files are valid
- Ensure generator is producing varied outputs
- Check for numerical stability issues

---

### Issue: Plots look wrong

**Possible causes:**
- Very different scales (e.g., real data in [0,1], generated in [0,100])
- Different sequence lengths
- Corrupted data

**Solution:**
- Verify data preprocessing is consistent
- Check sequence lengths match
- Inspect raw data files

---

## Best Practices

1. **Run analysis frequently during training**
   - Every 10-50 epochs
   - Track metric evolution
   - Detect problems early

2. **Focus on multiple metrics**
   - Don't rely on single metric
   - Use KS + Wasserstein + JS for robust assessment
   - Check both distribution and temporal properties

3. **Visualize early and often**
   - Plots reveal issues metrics might miss
   - Q-Q plots great for detecting specific distribution differences
   - Temporal plots catch unrealistic dynamics

4. **Compare across checkpoints**
   - Track improvement over training
   - Identify best checkpoint
   - Detect overfitting or mode collapse

5. **Save all analysis results**
   - Keep records for comparison
   - Document successful configurations
   - Build intuition for your data

---

## References

- **KS Test**: [Wikipedia - Kolmogorov-Smirnov test](https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test)
- **Wasserstein Distance**: [Wikipedia - Wasserstein metric](https://en.wikipedia.org/wiki/Wasserstein_metric)
- **JS Divergence**: [Wikipedia - Jensen-Shannon divergence](https://en.wikipedia.org/wiki/Jensen%E2%80%93Shannon_divergence)
- **Q-Q Plots**: [Wikipedia - Q-Q plot](https://en.wikipedia.org/wiki/Q%E2%80%93Q_plot)

---

**Questions or issues?** Open an issue on the GitHub repository.
