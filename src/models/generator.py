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

        # Output projection layers for Gaussian distribution
        self.mean_linear = nn.Linear(hidden_dim, 1)
        self.logvar_linear = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor):
        """
        Teacher-forcing pass for pretraining.
        Args:
            x: [batch, seq_len, 1] tensor of numerical values
        Returns:
            outputs: [batch, seq_len, 1] - sampled values
            means: [batch, seq_len, 1] - predicted means
            logvars: [batch, seq_len, 1] - predicted log variances
        """
        batch_size = x.size(0)
        device = x.device

        # Initialize hidden and cell states
        h = torch.zeros(batch_size, self.hidden_dim, device=device)
        c = torch.zeros(batch_size, self.hidden_dim, device=device)

        outputs_time = []
        means_time = []
        logvars_time = []

        for t in range(self.seq_len):
            h, c = self.lstm_cell(x[:, t, :], (h, c))

            # Get mean and log variance for Gaussian distribution
            mean = self.mean_linear(h)  # [B, 1]
            logvar = self.logvar_linear(h)  # [B, 1]

            # Clamp logvar for numerical stability
            logvar = torch.clamp(logvar, min=-10, max=2)

            # Reparameterization trick: sample = mean + std * epsilon
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            sample = mean + std * eps

            outputs_time.append(sample.unsqueeze(1))  # [B, 1, 1]
            means_time.append(mean.unsqueeze(1))
            logvars_time.append(logvar.unsqueeze(1))

        # Concatenate along time dimension [B, T, 1]
        outputs = torch.cat(outputs_time, dim=1)
        means = torch.cat(means_time, dim=1)
        logvars = torch.cat(logvars_time, dim=1)

        return outputs, means, logvars

    def generate(self, batch_size: int, requires_grad: bool = False):
        """
        Generate sequences without teacher forcing using Gaussian sampling.
        Args:
            batch_size: number of sequences to generate
            requires_grad: if True, maintain gradients for policy gradient training
        Returns:
            samples: [batch_size, seq_len, 1] tensor of generated values
            means: [batch_size, seq_len, 1] tensor of predicted means (if requires_grad)
            logvars: [batch_size, seq_len, 1] tensor of predicted log variances (if requires_grad)
        """
        device = next(self.parameters()).device
        h = torch.zeros(batch_size, self.hidden_dim, device=device)
        c = torch.zeros(batch_size, self.hidden_dim, device=device)

        # Start with the start value
        inp = torch.full((batch_size, 1), self.start_value, dtype=torch.float, device=device)
        samples = []
        means = []
        logvars = []

        if requires_grad:
            # Generate with gradients for policy gradient training
            for _ in range(self.seq_len):
                h, c = self.lstm_cell(inp, (h, c))

                # Get mean and log variance
                mean = self.mean_linear(h)
                logvar = self.logvar_linear(h)
                logvar = torch.clamp(logvar, min=-10, max=2)

                # Reparameterization trick with temperature
                std = torch.exp(0.5 * logvar) * self.temperature
                eps = torch.randn_like(std)
                sample = mean + std * eps

                inp = sample
                samples.append(sample.unsqueeze(1))
                means.append(mean.unsqueeze(1))
                logvars.append(logvar.unsqueeze(1))

            return (torch.cat(samples, dim=1),
                    torch.cat(means, dim=1),
                    torch.cat(logvars, dim=1))
        else:
            # Generate without gradients for sampling/evaluation
            with torch.no_grad():
                for _ in range(self.seq_len):
                    h, c = self.lstm_cell(inp, (h, c))

                    # Get mean and log variance
                    mean = self.mean_linear(h)
                    logvar = self.logvar_linear(h)
                    logvar = torch.clamp(logvar, min=-10, max=2)

                    # Sample with temperature
                    std = torch.exp(0.5 * logvar) * self.temperature
                    eps = torch.randn_like(std)
                    sample = mean + std * eps

                    inp = sample
                    samples.append(sample.unsqueeze(1))

            # [B, T, 1]
            return torch.cat(samples, dim=1)

    def gaussian_log_prob(self, x: torch.Tensor, mean: torch.Tensor, logvar: torch.Tensor):
        """
        Compute log probability of x under Gaussian distribution N(mean, exp(logvar)).
        Args:
            x: [B, T, 1] sampled values
            mean: [B, T, 1] predicted means
            logvar: [B, T, 1] predicted log variances
        Returns:
            log_prob: [B, T, 1] log probabilities
        """
        var = torch.exp(logvar)
        log_prob = -0.5 * (torch.log(2 * torch.pi * var) + ((x - mean) ** 2) / var)
        return log_prob

    def pretrain_loss(self, outputs: torch.Tensor, target: torch.Tensor, means: torch.Tensor = None, logvars: torch.Tensor = None):
        """
        Loss for pretraining - combination of MSE and negative log likelihood.
        Args:
            outputs: [B, T, 1] sampled values
            target: [B, T, 1] target values
            means: [B, T, 1] predicted means
            logvars: [B, T, 1] predicted log variances
        Returns:
            loss: scalar loss value
        """
        # MSE loss on samples
        mse_loss = F.mse_loss(outputs, target, reduction='mean')

        # If means and logvars provided, add negative log likelihood
        if means is not None and logvars is not None:
            log_prob = self.gaussian_log_prob(target, means, logvars)
            nll_loss = -log_prob.mean()
            return mse_loss + 0.1 * nll_loss  # Weighted combination
        else:
            return mse_loss

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
