# SeqGAN: Sequence Generative Adversarial Networks

A PyTorch implementation of SeqGAN for sequence generation using adversarial training. This project generates synthetic bond yield data (or other time series) using SeqGAN with Monte Carlo Rollouts and policy gradients for continuous actions.

## Project Structure

```
SeqGAN-1/
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── generator_loader.py      # Data loader for generator
│   │   └── discriminator_loader.py  # Data loader for discriminator
│   ├── models/
│   │   ├── __init__.py
│   │   ├── generator.py             # LSTM-based generator
│   │   ├── discriminator.py         # CNN-based discriminator
│   │   └── rollout.py               # Rollout policy for MCTS
│   ├── utils/
│   │   ├── __init__.py
│   │   └── training_utils.py        # Training utilities
│   └── config.py                    # Configuration and hyperparameters
├── save/                            # Directory for saved models and data
├── main.py                          # Main training script
├── requirements.txt                 # Python dependencies
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

Create a training data file at `save/real_data.txt` with one sequence per line. Each sequence should contain space-separated integers (e.g., token IDs). Default sequence length is 20.

Example:
```
1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20
5 3 8 2 9 1 4 7 6 10 15 12 18 13 19 14 20 16 17 11
...
```

### Run Training

Train the complete SeqGAN model:
```bash
python main.py
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
- LSTM-based recurrent neural network
- Generates sequences token-by-token
- Uses teacher forcing during pretraining
- Free-running generation during adversarial training

### Discriminator
- CNN-based architecture with highway networks
- Classifies sequences as real or generated
- Multiple convolutional and fully-connected layers
- Dropout for regularization

### Rollout Policy
- Monte Carlo Tree Search for reward estimation
- Tracks generator with exponential moving average
- Provides intermediate rewards for policy gradient

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

## Requirements

- Python 3.7+
- PyTorch 2.0+
- NumPy 1.24+
- Pandas 2.0+

## License

MIT License

## References

Based on the SeqGAN paper:
- Yu, L., Zhang, W., Wang, J., & Yu, Y. (2017). SeqGAN: Sequence Generative Adversarial Nets with Policy Gradient. In AAAI.
