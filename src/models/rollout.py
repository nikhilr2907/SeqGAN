import torch
import torch.nn as nn
import numpy as np


class Rollout(nn.Module):
    """Rollout policy for Monte Carlo Tree Search in adversarial training."""

    def __init__(self, generator, update_rate):
        """
        Args:
            generator: Generator model with LSTM architecture
            update_rate: float in (0,1), controls how fast rollout net tracks generator
        """
        super().__init__()
        self.generator = generator
        self.update_rate = update_rate
        self.seq_len = generator.seq_len
        self.hidden_dim = generator.hidden_dim

        # Create LSTM cell matching generator's architecture
        self.lstm_cell = nn.LSTMCell(1, self.hidden_dim)

        # Create output layer matching generator's architecture
        self.output_linear = nn.Linear(self.hidden_dim, 1)

        # Initialize with generator's weights
        self.update_params()

    def _rollout(self, prefix, given_len):
        """
        Generate continuation of the prefix sequence.
        Args:
            prefix: [batch, seq_len, 1] tensor
            given_len: number of timesteps already generated
        Returns:
            samples: [batch, seq_len, 1] complete sequences
        """
        batch_size = prefix.size(0)
        device = prefix.device

        # Initialize hidden states
        h = torch.zeros(batch_size, self.hidden_dim, device=device)
        c = torch.zeros(batch_size, self.hidden_dim, device=device)

        # Process the given prefix
        for t in range(given_len):
            h, c = self.lstm_cell(prefix[:, t, :], (h, c))

        # Generate remaining sequence
        samples = [prefix[:, :given_len, :]]
        current = prefix[:, given_len-1:given_len, :]

        for t in range(given_len, self.seq_len):
            h, c = self.lstm_cell(current.squeeze(1), (h, c))
            next_value = self.output_linear(h).unsqueeze(1)  # [B, 1, 1]
            samples.append(next_value)
            current = next_value

        return torch.cat(samples, dim=1)

    def get_reward(self, input_x, rollout_num, discriminator):
        """
        Calculate rewards for each position in the sequence using Monte Carlo rollouts.
        Args:
            input_x: [batch, seq_len, 1] generated sequences
            rollout_num: number of rollout samples per position
            discriminator: trained discriminator model
        Returns:
            rewards: [batch, seq_len] array of rewards
        """
        batch_size = input_x.size(0)
        rewards = []

        with torch.no_grad():
            # Evaluate rewards for each prefix length
            for given_len in range(1, self.seq_len):
                rollout_rewards = []
                for n in range(rollout_num):
                    # Generate completion from this prefix
                    samples = self._rollout(input_x, given_len)

                    # Get discriminator score
                    scores, _ = discriminator(samples)
                    # Use probability of being real (positive class)
                    prob = torch.softmax(scores, dim=1)[:, 1]
                    rollout_rewards.append(prob.cpu().numpy())

                # Average over rollouts
                avg_reward = np.mean(rollout_rewards, axis=0)
                rewards.append(avg_reward)

            # Reward for complete sequence
            scores, _ = discriminator(input_x)
            prob = torch.softmax(scores, dim=1)[:, 1]
            rewards.append(prob.cpu().numpy())

        # Shape: [batch, seq_len]
        rewards = np.array(rewards).T
        return rewards

    def update_params(self):
        """Update rollout network weights based on generator using exponential moving average."""
        with torch.no_grad():
            # Update LSTM cell parameters
            for rollout_param, gen_param in zip(self.lstm_cell.parameters(),
                                                self.generator.lstm_cell.parameters()):
                rollout_param.data.copy_(
                    self.update_rate * rollout_param.data +
                    (1 - self.update_rate) * gen_param.data
                )

            # Update output layer parameters
            for rollout_param, gen_param in zip(self.output_linear.parameters(),
                                                self.generator.output_linear.parameters()):
                rollout_param.data.copy_(
                    self.update_rate * rollout_param.data +
                    (1 - self.update_rate) * gen_param.data
                )
