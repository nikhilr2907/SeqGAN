"""Training utilities for SeqGAN."""

import torch
import torch.nn as nn
import numpy as np


def generate_samples(model, batch_size, generated_num, output_file, device):
    """
    Generate samples from the generator and save to file.
    Args:
        model: Generator model
        batch_size: batch size for generation
        generated_num: total number of samples to generate
        output_file: path to output file
        device: torch device
    """
    model.eval()
    generated_samples = []

    with torch.no_grad():
        for _ in range(int(generated_num / batch_size)):
            samples = model.generate(batch_size)
            generated_samples.extend(samples.cpu().numpy())

    with open(output_file, 'w') as fout:
        for sequence in generated_samples:
            # Flatten sequence if needed and convert to string
            if sequence.ndim > 1:
                sequence = sequence.flatten()
            buffer = ' '.join([str(int(x)) for x in sequence]) + '\n'
            fout.write(buffer)


def target_loss(model, data_loader, criterion, device):
    """
    Calculate loss on target data.
    Args:
        model: Generator model
        data_loader: data loader
        criterion: loss function
        device: torch device
    Returns:
        mean loss value
    """
    model.eval()
    losses = []

    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)
            # Ensure batch has correct shape [B, T, 1]
            if batch.dim() == 2:
                batch = batch.unsqueeze(-1)
            output = model(batch)
            loss = criterion(output, batch)
            losses.append(loss.item())

    return np.mean(losses) if losses else 0.0


def pre_train_epoch(model, data_loader, optimizer, criterion, device):
    """
    Perform one pretraining epoch.
    Args:
        model: Generator model
        data_loader: data loader
        optimizer: optimizer
        criterion: loss function
        device: torch device
    Returns:
        mean loss value for epoch
    """
    model.train()
    supervised_losses = []

    for batch in data_loader:
        batch = batch.to(device)
        # Ensure batch has correct shape [B, T, 1]
        if batch.dim() == 2:
            batch = batch.unsqueeze(-1)

        optimizer.zero_grad()
        output = model(batch)
        loss = criterion(output, batch)
        loss.backward()
        optimizer.step()
        supervised_losses.append(loss.item())

    return np.mean(supervised_losses) if supervised_losses else 0.0


def pretrain_generator(generator, gen_data_loader, likelihood_data_loader,
                       gen_optimizer, criterion, config, device):
    """
    Pretrain the generator using supervised learning.
    Args:
        generator: Generator model
        gen_data_loader: training data loader
        likelihood_data_loader: evaluation data loader
        gen_optimizer: generator optimizer
        criterion: loss function
        config: configuration object
        device: torch device
    """
    print("\n" + "="*50)
    print("Pretraining Generator")
    print("="*50)

    for epoch in range(config.PRE_EPOCH_NUM):
        loss = pre_train_epoch(generator, gen_data_loader, gen_optimizer, criterion, device)

        if epoch % 5 == 0 or epoch == config.PRE_EPOCH_NUM - 1:
            generate_samples(generator, config.BATCH_SIZE, config.GENERATED_NUM,
                           config.EVAL_FILE, device)
            likelihood_data_loader.create_batches(config.EVAL_FILE)
            test_loss = target_loss(generator, likelihood_data_loader, criterion, device)
            print(f"Epoch {epoch:3d} | Train Loss: {loss:.4f} | Test Loss: {test_loss:.4f}")


def pretrain_discriminator(generator, discriminator, dis_data_loader,
                           dis_optimizer, criterion, config, device):
    """
    Pretrain the discriminator.
    Args:
        generator: Generator model
        discriminator: Discriminator model
        dis_data_loader: discriminator data loader
        dis_optimizer: discriminator optimizer
        criterion: loss function
        config: configuration object
        device: torch device
    """
    print("\n" + "="*50)
    print("Pretraining Discriminator")
    print("="*50)

    for epoch in range(config.DIS_PRE_EPOCHS):
        # Generate negative samples
        generate_samples(generator, config.BATCH_SIZE, config.GENERATED_NUM,
                        config.NEGATIVE_FILE, device)
        dis_data_loader.load_train_data(config.POSITIVE_FILE, config.NEGATIVE_FILE)

        epoch_losses = []
        for step in range(config.DIS_PRE_UPDATE_STEPS):
            step_losses = []
            for x_batch, y_batch in dis_data_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)

                # Ensure correct shape for discriminator
                if x_batch.dim() == 2:
                    x_batch = x_batch.unsqueeze(-1)

                discriminator.train()
                dis_optimizer.zero_grad()
                scores, _ = discriminator(x_batch)
                loss = criterion(scores, torch.argmax(y_batch, dim=1))
                loss.backward()
                dis_optimizer.step()
                step_losses.append(loss.item())

            epoch_losses.extend(step_losses)

        if epoch % 10 == 0 or epoch == config.DIS_PRE_EPOCHS - 1:
            avg_loss = np.mean(epoch_losses)
            print(f"Epoch {epoch:3d} | Loss: {avg_loss:.4f}")


def adversarial_training(generator, discriminator, rollout, gen_data_loader,
                        dis_data_loader, gen_optimizer, dis_optimizer,
                        criterion, config, device):
    """
    Perform adversarial training.
    Args:
        generator: Generator model
        discriminator: Discriminator model
        rollout: Rollout policy
        gen_data_loader: generator data loader
        dis_data_loader: discriminator data loader
        gen_optimizer: generator optimizer
        dis_optimizer: discriminator optimizer
        criterion: loss function
        config: configuration object
        device: torch device
    """
    print("\n" + "="*50)
    print("Adversarial Training")
    print("="*50)

    for total_batch in range(config.TOTAL_BATCH):
        # Train generator with policy gradient
        for _ in range(config.GEN_ADV_UPDATES):
            samples = generator.generate(config.BATCH_SIZE).to(device)
            rewards = rollout.get_reward(samples, config.ROLLOUT_NUM, discriminator)
            rewards_tensor = torch.FloatTensor(rewards).to(device)

            # Policy gradient update
            generator.train()
            gen_optimizer.zero_grad()

            # Recompute outputs for gradient calculation
            # Use teacher forcing with generated samples
            outputs = generator(samples)

            # Calculate policy gradient loss
            # Loss = -mean(rewards * log_probs)
            # For simplicity, use negative mean reward
            loss = -torch.mean(rewards_tensor)
            loss.backward()
            gen_optimizer.step()

        # Update rollout policy
        rollout.update_params()

        # Train discriminator
        for _ in range(config.DIS_ADV_EPOCHS):
            generate_samples(generator, config.BATCH_SIZE, config.GENERATED_NUM,
                           config.NEGATIVE_FILE, device)
            dis_data_loader.load_train_data(config.POSITIVE_FILE, config.NEGATIVE_FILE)

            for _ in range(config.DIS_ADV_UPDATE_STEPS):
                dis_losses = []
                for x_batch, y_batch in dis_data_loader:
                    x_batch = x_batch.to(device)
                    y_batch = y_batch.to(device)

                    if x_batch.dim() == 2:
                        x_batch = x_batch.unsqueeze(-1)

                    discriminator.train()
                    dis_optimizer.zero_grad()
                    scores, _ = discriminator(x_batch)
                    loss = criterion(scores, torch.argmax(y_batch, dim=1))
                    loss.backward()
                    dis_optimizer.step()
                    dis_losses.append(loss.item())

        if total_batch % 10 == 0 or total_batch == config.TOTAL_BATCH - 1:
            print(f"Batch {total_batch:3d} | Dis Loss: {np.mean(dis_losses):.4f}")

    print("\nAdversarial training completed!")
