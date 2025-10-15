import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    """LSTM-based generator for sequence generation."""

    def __init__(self,
                 hidden_dim: int,
                 seq_len: int,
                 start_value: float,
                 reward_gamma: float = 0.95,
                 temperature: float = 1.0,
                 grad_clip: float = 5.0):
        """
        Args:
            hidden_dim: hidden size of the LSTM cell
            seq_len: length of sequence to generate
            start_value: initial value to feed in at t=0 when sampling
            reward_gamma: discount factor for RL
            temperature: scaling factor for sampling
            grad_clip: max-norm for gradient clipping
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        self.start_value = start_value
        self.reward_gamma = reward_gamma
        self.temperature = temperature
        self.grad_clip = grad_clip

        # LSTM cell for recurrent processing
        self.lstm_cell = nn.LSTMCell(1, hidden_dim)

        # Output projection layer
        self.output_linear = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor):
        """
        Teacher-forcing pass for pretraining.
        Args:
            x: [batch, seq_len, 1] tensor of numerical values
        Returns:
            outputs: [batch, seq_len, 1]
        """
        batch_size = x.size(0)
        device = x.device

        # Initialize hidden and cell states
        h = torch.zeros(batch_size, self.hidden_dim, device=device)
        c = torch.zeros(batch_size, self.hidden_dim, device=device)

        outputs_time = []
        for t in range(self.seq_len):
            h, c = self.lstm_cell(x[:, t, :], (h, c))
            output = self.output_linear(h).unsqueeze(1)  # [B, 1, 1]
            outputs_time.append(output)

        # Concatenate along time dimension [B, T, 1]
        outputs = torch.cat(outputs_time, dim=1)
        return outputs

    def generate(self, batch_size: int):
        """
        Generate sequences without teacher forcing.
        Args:
            batch_size: number of sequences to generate
        Returns:
            samples: [batch_size, seq_len, 1] tensor of generated values
        """
        device = next(self.parameters()).device
        h = torch.zeros(batch_size, self.hidden_dim, device=device)
        c = torch.zeros(batch_size, self.hidden_dim, device=device)

        # Start with the start value
        inp = torch.full((batch_size, 1), self.start_value, dtype=torch.float, device=device)
        samples = []

        with torch.no_grad():
            for _ in range(self.seq_len):
                h, c = self.lstm_cell(inp, (h, c))
                output = self.output_linear(h) / self.temperature
                inp = output
                samples.append(output.unsqueeze(1))

        # [B, T, 1]
        return torch.cat(samples, dim=1)

    def pretrain_loss(self, outputs: torch.Tensor, target: torch.Tensor):
        """
        MSE loss for numerical values.
        Args:
            outputs: [B, T, 1]
            target: [B, T, 1]
        Returns:
            loss: scalar loss value
        """
        return F.mse_loss(outputs, target, reduction='mean')

    def pretrain_step(self, optimizer, outputs: torch.Tensor, target: torch.Tensor):
        """
        Perform a pretraining step with gradient clipping.
        Args:
            optimizer: PyTorch optimizer for the generator
            outputs: [B, T, 1] tensor of generated numerical values
            target: [B, T, 1] tensor of target numerical values
        Returns:
            loss: scalar loss value
        """
        # Compute the MSE loss
        loss = F.mse_loss(outputs, target, reduction='mean')

        # Zero the gradients
        optimizer.zero_grad()

        # Backpropagate the loss
        loss.backward()

        # Clip gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip)

        # Update the model parameters
        optimizer.step()

        return loss
