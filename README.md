# SeqGAN: Sequence Generative Adversarial Networks

A PyTorch implementation of SeqGAN for sequence generation using adversarial training. This project generates synthetic continuous sequences using SeqGAN with Gaussian policy gradients, Monte Carlo rollouts, and comprehensive distribution analysis tools.

## Project Structure

```
SeqGAN-1/
├── src/
│   ├── data/
│   │   ├── generator_loader.py      # Data loader for generator
│   │   └── discriminator_loader.py  # Data loader for discriminator
│   ├── models/
│   │   ├── generator.py             # LSTM-based Gaussian generator
│   │   ├── discriminator.py         # CNN-based discriminator
│   │   └── rollout.py               # Rollout policy for MCTS
│   ├── utils/
│   │   ├── training_utils.py        # Training utilities
│   │   ├── distribution_metrics.py  # Distribution analysis metrics
│   │   └── visualize_distributions.py  # Visualization tools
│   └── config.py                    # Configuration and hyperparameters
├── tests/                           # Test suite
├── save/                            # Directory for saved models and data
├── main.py                          # Main training script
├── analyze_distributions.py         # Distribution analysis tool
└── README.md                        # This file
```

## Installation

1. Navigate to the project directory:
```bash
cd SeqGAN-1
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Prepare Training Data

Create a training data file at `save/real_data.txt` with one sequence per line. Each sequence should contain space-separated numerical values. Default sequence length is 20.

**File format requirements:**
- One sequence per line
- Space-separated floating-point numbers
- All sequences must have the same length (default: 20)
- No headers or metadata

Example:
```
0.12 0.45 -0.67 0.89 1.23 -0.34 0.56 0.78 -0.12 0.34 ...
0.23 -0.56 0.78 0.12 -0.89 0.45 0.67 -0.23 0.91 0.34 ...
...
```

**Data locations:**
- Training data: `save/real_data.txt` (you must create this)
- Generated samples: `save/final_samples.txt` (created after training)
- Analysis output: `./analysis_results/` (created when you run analysis)

### Run Training

Train the complete SeqGAN model:
```bash
python main.py
```

### Analyze Generated Distributions

After training, analyze the quality of generated sequences:
```bash
python analyse_distributions.py \
    --real save/real_data.txt \
    --generated save/final_samples.txt
```

**Output files (saved to `./analysis_results/`):**
- `analysis_metrics.txt` - Comprehensive metrics report
- `analysis_distributions.png` - 6 distribution comparison plots
- `analysis_temporal.png` - 4 temporal analysis plots

**What each file shows:**
- **Metrics**: KS statistic, Wasserstein distance, JS divergence, autocorrelation, diversity
- **Distributions**: Histograms, KDE, Q-Q plots, box plots, CDF, statistics comparison
- **Temporal**: Sample sequences, mean trends, autocorrelation, distribution evolution

For detailed interpretation of metrics and plots, see [DISTRIBUTION_ANALYSIS.md](DISTRIBUTION_ANALYSIS.md).

**Custom output directory:**
```bash
python analyse_distributions.py \
    --real save/real_data.txt \
    --generated save/final_samples.txt \
    --output-dir ./my_analysis
```

### Command-line Options

- `--show-config`: Display configuration parameters and exit
- `--skip-gen-pretrain`: Skip generator pretraining phase
- `--skip-dis-pretrain`: Skip discriminator pretraining phase
- `--skip-adversarial`: Skip adversarial training phase

Examples:
```bash
# Show configuration
python main.py --show-config

# Skip pretraining phases
python main.py --skip-gen-pretrain --skip-dis-pretrain
```

## Configuration

Edit `src/config.py` to modify hyperparameters:

### Generator Hyperparameters
- `HIDDEN_DIM`: LSTM hidden dimension (default: 32)
- `SEQ_LENGTH`: Sequence length (default: 20)
- `PRE_EPOCH_NUM`: Generator pretraining epochs (default: 120)
- `GEN_LR`: Generator learning rate (default: 1e-3)

### Discriminator Hyperparameters
- `DIS_DROPOUT_KEEP_PROB`: Dropout keep probability (default: 0.75)
- `DIS_L2_REG_LAMBDA`: L2 regularization (default: 0.2)
- `DIS_LR`: Discriminator learning rate (default: 1e-3)

### Training Hyperparameters
- `BATCH_SIZE`: Batch size (default: 64)
- `TOTAL_BATCH`: Adversarial training batches (default: 200)
- `SEED`: Random seed (default: 88)

## Model Architecture

### Generator
- LSTM-based recurrent neural network with Gaussian output
- Outputs mean and log-variance for each timestep
- Uses reparameterization trick for gradient flow
- Proper policy gradients with REINFORCE algorithm

### Discriminator
- CNN-based architecture with highway networks
- Classifies sequences as real or generated
- Multiple convolutional and fully-connected layers
- Dropout for regularization

### Rollout Policy
- Monte Carlo Tree Search for reward estimation
- Tracks generator with exponential moving average
- Provides intermediate rewards for policy gradient
- Automatic device consistency with generator

## Training Process

1. **Generator Pretraining**: Train generator on real data using maximum likelihood estimation
2. **Discriminator Pretraining**: Train discriminator to distinguish real vs generated sequences
3. **Adversarial Training**: Alternate between:
   - Update generator using policy gradient with rewards from discriminator
   - Update discriminator on new generated samples

## Output

The training script saves:
- `save/generator.pt`: Trained generator model
- `save/discriminator.pt`: Trained discriminator model
- `save/final_samples.txt`: Generated samples after training
- `save/generator_sample.txt`: Intermediate generated samples
- `save/eval_file.txt`: Evaluation samples

## Testing

Run tests to verify the implementation:
```bash
# Test gradient flow
python tests/test_gradient_flow.py

# Test device consistency
python tests/test_gpu_usage.py

# Run all tests
for test in tests/test_*.py; do python "$test"; done
```

See `tests/README.md` for detailed test documentation.

## Requirements

- Python 3.7+
- PyTorch 2.0+
- NumPy 1.24+
- Matplotlib 3.0+ (for visualization)
- Seaborn 0.11+ (for visualization)
- SciPy 1.7+ (for distribution metrics)


## References

Based on the SeqGAN paper:
- Yu, L., Zhang, W., Wang, J., & Yu, Y. (2017). SeqGAN: Sequence Generative Adversarial Nets with Policy Gradient. In AAAI.
