"""Configuration and hyperparameters for SeqGAN."""

import os


class Config:
    """Configuration class for SeqGAN training."""

    # Generator hyperparameters
    HIDDEN_DIM = 32  # Hidden state dimension of LSTM cell
    SEQ_LENGTH = 20  # Sequence length
    START_TOKEN = 3.0  # Start token for generation (should match mean of data ~2.7-3.0)
    PRE_EPOCH_NUM = 120  # Pretraining epochs for generator
    TEMPERATURE = 1.0  # Temperature for sampling
    GRAD_CLIP = 5.0  # Gradient clipping threshold

    # Discriminator hyperparameters
    DIS_DROPOUT_KEEP_PROB = 0.75  # Dropout keep probability
    DIS_L2_REG_LAMBDA = 0.2  # L2 regularization coefficient

    # Training hyperparameters
    SEED = 88  # Random seed
    BATCH_SIZE = 64  # Batch size for training
    TOTAL_BATCH = 200  # Total adversarial training batches
    GENERATED_NUM = 10000  # Number of samples to generate

    # Learning rates
    GEN_LR = 1e-3  # Generator learning rate
    DIS_LR = 1e-3  # Discriminator learning rate

    # Rollout parameters
    ROLLOUT_UPDATE_RATE = 0.8  # Rollout network update rate
    ROLLOUT_NUM = 16  # Number of rollout samples

    # Pre-training parameters
    DIS_PRE_EPOCHS = 50  # Discriminator pretraining epochs
    DIS_PRE_UPDATE_STEPS = 3  # Discriminator updates per pretraining epoch

    # Adversarial training parameters
    GEN_ADV_UPDATES = 1  # Generator updates per adversarial batch
    DIS_ADV_EPOCHS = 5  # Discriminator epochs per adversarial batch
    DIS_ADV_UPDATE_STEPS = 3  # Discriminator updates per adversarial epoch

    # File paths
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    SAVE_DIR = os.path.join(BASE_DIR, 'save')
    POSITIVE_FILE = os.path.join(SAVE_DIR, 'real_data.txt')
    NEGATIVE_FILE = os.path.join(SAVE_DIR, 'generator_sample.txt')
    EVAL_FILE = os.path.join(SAVE_DIR, 'eval_file.txt')

    # Model checkpoint paths
    GEN_CHECKPOINT = os.path.join(SAVE_DIR, 'generator.pt')
    DIS_CHECKPOINT = os.path.join(SAVE_DIR, 'discriminator.pt')

    @classmethod
    def display(cls):
        """Display all configuration parameters."""
        print("=" * 50)
        print("SeqGAN Configuration")
        print("=" * 50)
        for attr in dir(cls):
            if not attr.startswith('_') and attr.isupper():
                print(f"{attr}: {getattr(cls, attr)}")
        print("=" * 50)
