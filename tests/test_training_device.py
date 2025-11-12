"""Test script to verify device consistency during training."""

import torch
from src.models import Generator, Discriminator, Rollout
from src.config import Config

def test_training_device_flow():
    """Test device consistency in full training flow."""
    print("Testing Training Device Flow...")
    print("="*50)

    config = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    # Initialize models
    print("\n1. Initializing models...")
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

    rollout = Rollout(generator, config.ROLLOUT_UPDATE_RATE)

    print(f"   Generator device: {next(generator.parameters()).device}")
    print(f"   Discriminator device: {next(discriminator.parameters()).device}")
    print(f"   Rollout device: {next(rollout.parameters()).device}")

    # Verify all on same device
    gen_device = next(generator.parameters()).device
    dis_device = next(discriminator.parameters()).device
    rollout_device = next(rollout.parameters()).device

    if gen_device == dis_device == rollout_device:
        print("   [OK] All models on same device")
    else:
        print("   [FAIL] Models on different devices!")
        return False

    # Test generator forward pass
    print("\n2. Testing generator forward pass...")
    batch_size = config.BATCH_SIZE
    x = torch.randn(batch_size, config.SEQ_LENGTH, 1).to(device)

    try:
        outputs, means, logvars = generator(x)
        print(f"   Outputs device: {outputs.device}")
        print(f"   [OK] Generator forward pass successful")
    except RuntimeError as e:
        if "device" in str(e).lower():
            print(f"   [FAIL] Device error: {e}")
            return False
        raise

    # Test generator.generate with requires_grad=True
    print("\n3. Testing generator.generate with gradients...")
    try:
        samples, means, logvars = generator.generate(batch_size, requires_grad=True)
        print(f"   Samples device: {samples.device}")
        print(f"   Samples requires_grad: {samples.requires_grad}")
        print(f"   [OK] Generation with gradients successful")
    except RuntimeError as e:
        if "device" in str(e).lower():
            print(f"   [FAIL] Device error: {e}")
            return False
        raise

    # Test discriminator
    print("\n4. Testing discriminator...")
    try:
        scores, features = discriminator(samples.detach())
        print(f"   Scores device: {scores.device}")
        print(f"   [OK] Discriminator forward pass successful")
    except RuntimeError as e:
        if "device" in str(e).lower():
            print(f"   [FAIL] Device error: {e}")
            return False
        raise

    # Test rollout.get_reward
    print("\n5. Testing rollout.get_reward...")
    try:
        rewards = rollout.get_reward(samples.detach(), rollout_num=2, discriminator=discriminator)
        print(f"   Rewards shape: {rewards.shape}")
        print(f"   Rewards range: [{rewards.min():.3f}, {rewards.max():.3f}]")
        print(f"   [OK] Rollout reward computation successful")
    except RuntimeError as e:
        if "device" in str(e).lower():
            print(f"   [FAIL] Device error: {e}")
            return False
        raise

    # Test policy gradient computation
    print("\n6. Testing policy gradient computation...")
    try:
        # Generate with gradients
        samples, means, logvars = generator.generate(batch_size, requires_grad=True)

        # Get rewards (detached)
        with torch.no_grad():
            rewards_np = rollout.get_reward(samples.detach(), rollout_num=2, discriminator=discriminator)
        rewards_tensor = torch.FloatTensor(rewards_np).to(device)

        # Compute log probabilities
        log_probs = generator.gaussian_log_prob(samples, means, logvars)

        print(f"   Samples device: {samples.device}")
        print(f"   Rewards device: {rewards_tensor.device}")
        print(f"   Log probs device: {log_probs.device}")

        # Compute loss
        advantages = rewards_tensor - rewards_tensor.mean()
        advantages = advantages.unsqueeze(-1)
        pg_loss = -(log_probs * advantages).mean()

        print(f"   Policy gradient loss: {pg_loss.item():.4f}")
        print(f"   [OK] Policy gradient computation successful")
    except RuntimeError as e:
        if "device" in str(e).lower():
            print(f"   [FAIL] Device error: {e}")
            return False
        raise

    # Test backward pass
    print("\n7. Testing backward pass...")
    try:
        generator.zero_grad()
        pg_loss.backward()

        # Check gradients exist and are on correct device
        grad_count = 0
        for name, param in generator.named_parameters():
            if param.grad is not None:
                if param.grad.device != device:
                    print(f"   [FAIL] Gradient for {name} on wrong device: {param.grad.device}")
                    return False
                grad_count += 1

        print(f"   [OK] Backward pass successful, {grad_count} parameters with gradients")
    except RuntimeError as e:
        if "device" in str(e).lower():
            print(f"   [FAIL] Device error: {e}")
            return False
        raise

    # Test rollout.update_params
    print("\n8. Testing rollout.update_params...")
    try:
        rollout.update_params()

        # Verify rollout still on correct device
        rollout_device_after = next(rollout.parameters()).device
        if rollout_device_after != device:
            print(f"   [FAIL] Rollout moved to wrong device: {rollout_device_after}")
            return False

        print(f"   Rollout device after update: {rollout_device_after}")
        print(f"   [OK] Rollout update successful")
    except RuntimeError as e:
        if "device" in str(e).lower():
            print(f"   [FAIL] Device error: {e}")
            return False
        raise

    print("\n" + "="*50)
    print("[SUCCESS] All training device flow tests passed!")
    print("="*50)

    return True

if __name__ == "__main__":
    success = test_training_device_flow()
    exit(0 if success else 1)
