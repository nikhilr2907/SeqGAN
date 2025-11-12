"""Test script to verify LSTM cell device consistency in rollout."""

import torch
from src.models import Generator, Rollout

def test_lstm_device_consistency():
    """Test that all rollout components are on the same device."""
    print("Testing LSTM Cell Device Consistency...")
    print("="*50)

    # Test with CPU first
    print("\n=== Testing with CPU ===")
    generator_cpu = Generator(
        hidden_dim=32,
        seq_len=20,
        start_value=0.0,
        temperature=1.0,
        grad_clip=5.0
    )

    device_cpu = torch.device("cpu")
    generator_cpu = generator_cpu.to(device_cpu)

    print(f"Generator device: {next(generator_cpu.parameters()).device}")

    # Initialize rollout
    rollout_cpu = Rollout(generator_cpu, update_rate=0.8)

    # Check all components
    print("\nChecking rollout components on CPU:")
    for name, param in rollout_cpu.named_parameters():
        print(f"  {name}: {param.device}")

    # Verify LSTM cell specifically
    lstm_weight_ih = rollout_cpu.lstm_cell.weight_ih.device
    lstm_weight_hh = rollout_cpu.lstm_cell.weight_hh.device
    mean_weight = rollout_cpu.mean_linear.weight.device

    print(f"\nLSTM weight_ih device: {lstm_weight_ih}")
    print(f"LSTM weight_hh device: {lstm_weight_hh}")
    print(f"Mean linear weight device: {mean_weight}")

    all_cpu = all(
        param.device.type == 'cpu'
        for param in rollout_cpu.parameters()
    )

    if all_cpu:
        print("[OK] All parameters on CPU")
    else:
        print("[FAIL] Some parameters not on CPU!")
        return False

    # Test with CUDA if available
    if torch.cuda.is_available():
        print("\n=== Testing with CUDA ===")
        generator_cuda = Generator(
            hidden_dim=32,
            seq_len=20,
            start_value=0.0,
            temperature=1.0,
            grad_clip=5.0
        )

        device_cuda = torch.device("cuda")
        generator_cuda = generator_cuda.to(device_cuda)

        print(f"Generator device: {next(generator_cuda.parameters()).device}")

        # Initialize rollout
        rollout_cuda = Rollout(generator_cuda, update_rate=0.8)

        # Check all components
        print("\nChecking rollout components on CUDA:")
        for name, param in rollout_cuda.named_parameters():
            print(f"  {name}: {param.device}")

        all_cuda = all(
            param.device.type == 'cuda'
            for param in rollout_cuda.parameters()
        )

        if all_cuda:
            print("[OK] All parameters on CUDA")
        else:
            print("[FAIL] Some parameters not on CUDA!")
            return False

    # Test forward pass to ensure no device mismatch errors
    print("\n=== Testing Forward Pass ===")
    batch_size = 4

    # Generate samples on the correct device
    samples = generator_cpu.generate(batch_size, requires_grad=False)
    print(f"Generated samples device: {samples.device}")

    # Try rollout
    try:
        rollout_samples = rollout_cpu._rollout(samples, given_len=10)
        print(f"Rollout samples device: {rollout_samples.device}")
        print("[OK] Forward pass successful!")
    except RuntimeError as e:
        if "device" in str(e).lower():
            print(f"[FAIL] Device mismatch error: {e}")
            return False
        else:
            raise

    # Test after update_params
    print("\n=== Testing After update_params() ===")
    rollout_cpu.update_params()

    print("Checking rollout components after update:")
    for name, param in rollout_cpu.named_parameters():
        if param.device.type != 'cpu':
            print(f"  [FAIL] {name}: {param.device} (expected CPU)")
            return False

    print("[OK] All parameters still on correct device after update")

    # Test forward pass again
    try:
        samples2 = generator_cpu.generate(batch_size, requires_grad=False)
        rollout_samples2 = rollout_cpu._rollout(samples2, given_len=10)
        print("[OK] Forward pass after update successful!")
    except RuntimeError as e:
        if "device" in str(e).lower():
            print(f"[FAIL] Device mismatch error after update: {e}")
            return False
        else:
            raise

    print("\n" + "="*50)
    print("[SUCCESS] All LSTM device tests passed!")
    print("="*50)

    return True

if __name__ == "__main__":
    success = test_lstm_device_consistency()
    exit(0 if success else 1)
