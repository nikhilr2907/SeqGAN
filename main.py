import os
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.config import Config
from src.data import GenDataLoader, DisDataLoader
from src.models import Generator, Discriminator, Rollout
from src.utils import (
    generate_samples,
    pretrain_generator,
    pretrain_discriminator,
    adversarial_training
)


def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def setup_device():
    """Setup and return the appropriate device (CPU/GPU)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    return device


def initialize_models(config, device):
    """Initialize generator and discriminator models."""
    generator = Generator(
        hidden_dim=config.HIDDEN_DIM,
        seq_len=config.SEQ_LENGTH,
        start_value=config.START_TOKEN,
        temperature=config.TEMPERATURE,
        grad_clip=config.GRAD_CLIP
    ).to(device)

    discriminator = Discriminator(
        sequence_length=config.SEQ_LENGTH,
        data_size=1,
        l2_reg_lambda=config.DIS_L2_REG_LAMBDA
    ).to(device)

    print(f"\nGenerator parameters: {sum(p.numel() for p in generator.parameters()):,}")
    print(f"Discriminator parameters: {sum(p.numel() for p in discriminator.parameters()):,}")

    return generator, discriminator


def initialize_data_loaders(config):
    """Initialize data loaders for training."""
    gen_data_loader = GenDataLoader(config.BATCH_SIZE)
    likelihood_data_loader = GenDataLoader(config.BATCH_SIZE)
    dis_data_loader = DisDataLoader(config.BATCH_SIZE)

    # Check if positive file exists
    if not os.path.exists(config.POSITIVE_FILE):
        print(f"\nWarning: Training data file not found at {config.POSITIVE_FILE}")
        print("Please create a training data file with sequences (one per line).")
        print("Each sequence should contain space-separated integers.")
        return None, None, None

    # Load initial data
    gen_data_loader.create_batches(config.POSITIVE_FILE)
    print(f"\nLoaded {len(gen_data_loader)} batches of training data")

    return gen_data_loader, likelihood_data_loader, dis_data_loader


def main(args):
    """Main training function."""
    # Display configuration
    config = Config()
    if args.show_config:
        config.display()
        return

    # Setup
    set_seed(config.SEED)
    device = setup_device()

    # Create save directory if it doesn't exist
    os.makedirs(config.SAVE_DIR, exist_ok=True)

    # Initialize data loaders
    gen_data_loader, likelihood_data_loader, dis_data_loader = initialize_data_loaders(config)
    if gen_data_loader is None:
        return

    # Initialize models
    generator, discriminator = initialize_models(config, device)

    # Initialize optimizers
    gen_optimizer = optim.Adam(generator.parameters(), lr=config.GEN_LR)
    dis_optimizer = optim.Adam(
        discriminator.parameters(),
        lr=config.DIS_LR,
        weight_decay=config.DIS_L2_REG_LAMBDA
    )

    # Loss function
    criterion = nn.MSELoss()  # For generator pretraining
    dis_criterion = nn.CrossEntropyLoss()  # For discriminator

    # # Pretrain generator
    # if not args.skip_gen_pretrain:
    #     pretrain_generator(
    #         generator, gen_data_loader, likelihood_data_loader,
    #         gen_optimizer, criterion, config, device
    #     )
    #     # Save generator checkpoint
    #     torch.save(generator.state_dict(), config.GEN_CHECKPOINT)
    #     print(f"\nGenerator checkpoint saved to {config.GEN_CHECKPOINT}")
    # else:
    #     print("\nSkipping generator pretraining")
    print("\nSkipping generator pretraining - going straight to adversarial training")

    # # Pretrain discriminator
    # if not args.skip_dis_pretrain:
    #     pretrain_discriminator(
    #         generator, discriminator, dis_data_loader,
    #         dis_optimizer, dis_criterion, config, device
    #     )
    #     # Save discriminator checkpoint
    #     torch.save(discriminator.state_dict(), config.DIS_CHECKPOINT)
    #     print(f"\nDiscriminator checkpoint saved to {config.DIS_CHECKPOINT}")
    # else:
    #     print("\nSkipping discriminator pretraining")

    # Initialize rollout policy
    rollout = Rollout(generator, config.ROLLOUT_UPDATE_RATE)

    # Adversarial training
    if not args.skip_adversarial:
        adversarial_training(
            generator, discriminator, rollout,
            gen_data_loader, dis_data_loader,
            gen_optimizer, dis_optimizer,
            dis_criterion, config, device
        )

        # Save final models
        torch.save(generator.state_dict(), config.GEN_CHECKPOINT)
        torch.save(discriminator.state_dict(), config.DIS_CHECKPOINT)
        print(f"\nFinal models saved to {config.SAVE_DIR}")
    else:
        print("\nSkipping adversarial training")

    # Generate final samples
    print("\nGenerating final samples...")
    final_samples_file = os.path.join(config.SAVE_DIR, 'final_samples.txt')
    generate_samples(generator, config.BATCH_SIZE, config.GENERATED_NUM,
                    final_samples_file, device)
    print(f"Final samples saved to {final_samples_file}")

    print("\n" + "="*50)
    print("Training completed successfully!")
    print("="*50)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SeqGAN Training')
    parser.add_argument('--show-config', action='store_true',
                       help='Display configuration and exit')
    parser.add_argument('--skip-gen-pretrain', action='store_true',
                       help='Skip generator pretraining')
    parser.add_argument('--skip-dis-pretrain', action='store_true',
                       help='Skip discriminator pretraining')
    parser.add_argument('--skip-adversarial', action='store_true',
                       help='Skip adversarial training')

    args = parser.parse_args()
    main(args)
