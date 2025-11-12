"""Test script to verify all operations run on GPU when CUDA is available."""

import torch
from src.models import Generator, Discriminator, Rollout
from src.config import Config
from src.data import GenDataLoader, DisDataLoader

def check_tensor_device(tensor, expected_device, name):
    """Helper to check if tensor is on expected device."""
    if tensor.device.type == expected_device.type:
        print(f"   [OK] {name}: {tensor.device}")
        return True
    else:
        print(f"   [FAIL] {name}: {tensor.device} (expected {expected_device})")
        return False

def test_gpu_usage():
    """Comprehensive test of GPU usage throughout training."""
    print("Testing GPU Usage...")
    print("="*50)

    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("\nCUDA not available. Testing with CPU instead.")
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")
        print(f"\nCUDA available! Using GPU: {torch.cuda.get_device_name(0)}")

    config = Config()
    all_passed = True

    # 1. Test model initialization
    print("\n1. Testing Model Initialization...")
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

    # Check all model parameters
    gen_device = next(generator.parameters()).device
    dis_device = next(discriminator.parameters()).device
    rollout_device = next(rollout.parameters()).device

    all_passed &= check_tensor_device(torch.zeros(1).to(gen_device), device, "Generator")
    all_passed &= check_tensor_device(torch.zeros(1).to(dis_device), device, "Discriminator")
    all_passed &= check_tensor_device(torch.zeros(1).to(rollout_device), device, "Rollout")

    # 2. Test data loading and device transfer
    print("\n2. Testing Data Loading...")
    batch_size = config.BATCH_SIZE

    # Simulate data loader behavior
    batch = torch.randn(batch_size, config.SEQ_LENGTH, 1).to(device)
    all_passed &= check_tensor_device(batch, device, "Data batch")

    # 3. Test generator forward pass
    print("\n3. Testing Generator Forward Pass...")
    outputs, means, logvars = generator(batch)
    all_passed &= check_tensor_device(outputs, device, "Generator outputs")
    all_passed &= check_tensor_device(means, device, "Generator means")
    all_passed &= check_tensor_device(logvars, device, "Generator logvars")

    # 4. Test generator.generate
    print("\n4. Testing Generator.generate (with gradients)...")
    samples, means, logvars = generator.generate(batch_size, requires_grad=True)
    all_passed &= check_tensor_device(samples, device, "Generated samples")
    all_passed &= check_tensor_device(means, device, "Generated means")
    all_passed &= check_tensor_device(logvars, device, "Generated logvars")
    print(f"   Samples require_grad: {samples.requires_grad}")

    # 5. Test discriminator
    print("\n5. Testing Discriminator...")
    scores, features = discriminator(samples.detach())
    all_passed &= check_tensor_device(scores, device, "Discriminator scores")
    all_passed &= check_tensor_device(features, device, "Discriminator features")

    # 6. Test rollout
    print("\n6. Testing Rollout...")
    rewards = rollout.get_reward(samples.detach(), rollout_num=2, discriminator=discriminator)
    print(f"   Rewards (numpy array) shape: {rewards.shape}")

    # Convert rewards to tensor on device
    rewards_tensor = torch.FloatTensor(rewards).to(device)
    all_passed &= check_tensor_device(rewards_tensor, device, "Rewards tensor")

    # 7. Test policy gradient computation
    print("\n7. Testing Policy Gradient Computation...")
    log_probs = generator.gaussian_log_prob(samples, means, logvars)
    all_passed &= check_tensor_device(log_probs, device, "Log probabilities")

    advantages = rewards_tensor - rewards_tensor.mean()
    advantages = advantages.unsqueeze(-1)
    all_passed &= check_tensor_device(advantages, device, "Advantages")

    pg_loss = -(log_probs * advantages).mean()
    all_passed &= check_tensor_device(pg_loss, device, "Policy gradient loss")

    # 8. Test backward pass
    print("\n8. Testing Backward Pass...")
    generator.zero_grad()
    pg_loss.backward()

    # Check gradients are on correct device
    for name, param in generator.named_parameters():
        if param.grad is not None:
            if param.grad.device.type != device.type:
                print(f"   [FAIL] Gradient for {name} on wrong device: {param.grad.device}")
                all_passed = False
            break  # Just check first gradient
    else:
        print("   [FAIL] No gradients found!")
        all_passed = False

    if all_passed:
        print("   [OK] All gradients on correct device")

    # 9. Test discriminator training step
    print("\n9. Testing Discriminator Training Step...")
    x_batch = torch.randn(batch_size, config.SEQ_LENGTH, 1).to(device)
    y_batch = torch.FloatTensor([[0, 1] for _ in range(batch_size)]).to(device)

    all_passed &= check_tensor_device(x_batch, device, "Discriminator input batch")
    all_passed &= check_tensor_device(y_batch, device, "Discriminator label batch")

    x_batch = x_batch.unsqueeze(-1) if x_batch.dim() == 2 else x_batch
    scores, _ = discriminator(x_batch)
    criterion = torch.nn.CrossEntropyLoss()
    dis_loss = criterion(scores, torch.argmax(y_batch, dim=1))

    all_passed &= check_tensor_device(dis_loss, device, "Discriminator loss")

    # 10. Test memory usage (if CUDA)
    if device.type == "cuda":
        print("\n10. Testing GPU Memory Usage...")
        print(f"   Allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
        print(f"   Cached: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")

    # Summary
    print("\n" + "="*50)
    if all_passed:
        print(f"[SUCCESS] All operations on {device.type.upper()}!")
        if device.type == "cuda":
            print("Your training will run on GPU! 🚀")
    else:
        print(f"[WARNING] Some operations not on {device.type.upper()}")
    print("="*50)

    return all_passed

if __name__ == "__main__":
    success = test_gpu_usage()
    exit(0 if success else 1)
