"""Test script to verify rollout device consistency."""

import torch
from src.models import Generator, Rollout, Discriminator

def test_rollout_device():
    """Test that rollout stays on the same device as generator."""
    print("Testing Rollout Device Consistency...")
    print("="*50)

    # Initialize generator
    generator = Generator(
        hidden_dim=32,
        seq_len=20,
        start_value=0.0,
        temperature=1.0,
        grad_clip=5.0
    )

    # Initialize discriminator
    discriminator = Discriminator(
        sequence_length=20,
        data_size=1,
        l2_reg_lambda=0.0
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n1. Testing with device: {device}")

    generator = generator.to(device)
    discriminator = discriminator.to(device)

    # Initialize rollout
    print("\n2. Initializing rollout...")
    rollout = Rollout(generator, update_rate=0.8)

    # Check device consistency
    gen_device = next(generator.parameters()).device
    rollout_device = next(rollout.parameters()).device

    print(f"   Generator device: {gen_device}")
    print(f"   Rollout device: {rollout_device}")

    if gen_device == rollout_device:
        print("   [OK] Devices match!")
    else:
        print(f"   [FAIL] Device mismatch!")
        return False

    # Test rollout generation
    print("\n3. Testing rollout generation...")
    batch_size = 4
    samples = generator.generate(batch_size, requires_grad=False)
    print(f"   Generated samples shape: {samples.shape}")
    print(f"   Samples device: {samples.device}")

    # Test get_reward
    print("\n4. Testing rollout reward computation...")
    try:
        rewards = rollout.get_reward(samples, rollout_num=2, discriminator=discriminator)
        print(f"   Rewards shape: {rewards.shape}")
        print(f"   Rewards range: [{rewards.min():.3f}, {rewards.max():.3f}]")
        print("   [OK] Reward computation successful!")
    except Exception as e:
        print(f"   [FAIL] Reward computation failed: {e}")
        return False

    # Test update_params
    print("\n5. Testing rollout parameter update...")
    rollout.update_params()

    rollout_device_after = next(rollout.parameters()).device
    print(f"   Rollout device after update: {rollout_device_after}")

    if rollout_device_after == gen_device:
        print("   [OK] Device consistency maintained after update!")
    else:
        print(f"   [FAIL] Device changed after update!")
        return False

    # Verify architecture matches
    print("\n6. Verifying architecture consistency...")
    has_mean = hasattr(rollout, 'mean_linear')
    has_logvar = hasattr(rollout, 'logvar_linear')
    has_old_output = hasattr(rollout, 'output_linear')

    print(f"   Rollout has mean_linear: {has_mean}")
    print(f"   Rollout has logvar_linear: {has_logvar}")
    print(f"   Rollout has old output_linear: {has_old_output}")

    if has_mean and has_logvar and not has_old_output:
        print("   [OK] Architecture matches generator!")
    else:
        print("   [FAIL] Architecture mismatch!")
        return False

    print("\n" + "="*50)
    print("[SUCCESS] All rollout tests passed!")
    print("="*50)

    return True

if __name__ == "__main__":
    success = test_rollout_device()
    exit(0 if success else 1)
