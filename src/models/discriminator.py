import torch
import torch.nn as nn
import torch.nn.functional as F


class Highway(nn.Module):
    """Highway network layer for gated information flow."""

    def __init__(self, size, num_layers=1, bias=-2.0, activation=F.relu):
        super().__init__()
        self.num_layers = num_layers
        self.bias = bias
        self.activation = activation

        # Linear transformations and gates for each layer
        self.linears = nn.ModuleList([nn.Linear(size, size) for _ in range(num_layers)])
        self.gates = nn.ModuleList([nn.Linear(size, size) for _ in range(num_layers)])

    def forward(self, x):
        """
        Args:
            x: [batch, size]
        Returns:
            output: [batch, size]
        """
        for lin, gate in zip(self.linears, self.gates):
            g = self.activation(lin(x))
            t = torch.sigmoid(gate(x) + self.bias)
            x = t * g + (1 - t) * x
        return x


class Discriminator(nn.Module):
    """CNN-based discriminator for sequence classification."""

    def __init__(self, sequence_length, data_size, l2_reg_lambda=0.0):
        """
        Args:
            sequence_length: length of input sequences
            data_size: number of channels/features per timestep
            l2_reg_lambda: coefficient for L2 penalty (use as weight_decay in optimizer)
        """
        super().__init__()
        self.l2_reg_lambda = l2_reg_lambda

        # Two 2D convolutional layers (input shape: [B, 1, L, D])
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=3, padding=1)

        # Compute flattened size after conv2
        flat_size = 32 * sequence_length * data_size

        # Highway network
        self.highway = Highway(flat_size, num_layers=2)

        # Six fully-connected layers
        self.fc1 = nn.Linear(flat_size, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, 64)
        self.fc6 = nn.Linear(64, 32)

        # 1x1 convolution and pooling
        self.conv_pool_conv = nn.Conv2d(32, 64, kernel_size=1)
        self.pool = nn.MaxPool2d(kernel_size=1)

        # Final output layer
        self.output = nn.Linear(64, 2)

    def forward(self, x, dropout_keep_prob=0.75):
        """
        Args:
            x: [batch, sequence_length, data_size] or [batch, 1, sequence_length, data_size]
            dropout_keep_prob: keep probability for dropout
        Returns:
            scores: [batch, 2] logits
            predictions: [batch] predicted class indices
        """
        # Ensure input has correct shape [B, 1, L, D]
        if x.dim() == 3:
            x = x.unsqueeze(1)

        # Convolutional stack
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        # Flatten
        x = x.view(x.size(0), -1)

        # Highway network with dropout
        x = self.highway(x)
        x = F.dropout(x, p=1-dropout_keep_prob, training=self.training)

        # Six fully-connected layers with dropout
        for fc in (self.fc1, self.fc2, self.fc3, self.fc4, self.fc5, self.fc6):
            x = F.relu(fc(x))
            x = F.dropout(x, p=1-dropout_keep_prob, training=self.training)

        # Reshape for convolution-pooling
        x = x.unsqueeze(-1).unsqueeze(-1)  # [B, 32, 1, 1]
        x = F.relu(self.conv_pool_conv(x))  # [B, 64, 1, 1]
        x = self.pool(x)  # [B, 64, 1, 1]
        x = x.view(x.size(0), -1)  # [B, 64]

        # Final scores and predictions
        scores = self.output(x)  # [B, 2]
        predictions = torch.argmax(scores, dim=1)

        return scores, predictions
