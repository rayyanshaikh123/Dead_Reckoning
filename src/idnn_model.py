"""
IDNN (Input Delay Neural Network) model for smartphone INS denoising.

An IDNN uses tapped delay lines on the input — instead of just the current
timestep, it takes a window of [x(t), x(t-1), …, x(t-d)] as a flattened
input to a feedforward network.  This gives temporal context without
recurrence.

The model learns to map noisy smartphone sensor readings to clean vehicle
ground-truth values (speed, acceleration).
"""

import torch
import torch.nn as nn


class IDNN(nn.Module):
    """
    Input Delay Neural Network.

    Takes a flattened window of `(delay_taps + 1) * n_features` inputs
    and predicts `n_outputs` target values.

    Architecture:
        Input → Linear → ReLU → Dropout →
        Linear → ReLU → Dropout →
        Linear → ReLU →
        Linear → Output
    """

    def __init__(
        self,
        n_features: int,
        delay_taps: int,
        n_outputs: int = 1,
        hidden_sizes: tuple = (128, 64, 32),
        dropout: float = 0.2,
    ):
        super().__init__()

        self.n_features = n_features
        self.delay_taps = delay_taps
        self.n_outputs = n_outputs

        input_dim = n_features * (delay_taps + 1)

        layers = []
        prev_dim = input_dim

        for i, h in enumerate(hidden_sizes):
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            if i < len(hidden_sizes) - 1:
                layers.append(nn.Dropout(dropout))
            prev_dim = h

        layers.append(nn.Linear(prev_dim, n_outputs))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch, (delay_taps+1) * n_features)

        Returns:
            Tensor of shape (batch, n_outputs)
        """
        return self.network(x)

