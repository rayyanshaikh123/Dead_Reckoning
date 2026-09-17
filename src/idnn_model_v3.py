"""
Enhanced IDNN v3  --  Input Delay Neural Network with BatchNorm.

Same proven feedforward-over-delay-taps architecture as v1, enhanced with:
  - BatchNorm after each hidden layer (faster training, better generalisation)
  - Wider hidden layers to accommodate extra jerk features
  - Configurable architecture via constructor args

No recurrence, no memory poisoning.
"""

import torch
import torch.nn as nn


class IDNNv3(nn.Module):
    """
    Input Delay Neural Network v3.

    Input : flattened window (delay_taps + 1) * n_features
    Output: n_outputs (default 1 = speed in m/s)
    """

    def __init__(
        self,
        n_features: int,
        delay_taps: int,
        n_outputs: int = 1,
        hidden_sizes: tuple = (256, 128, 64),
        dropout: float = 0.2,
    ):
        super().__init__()

        self.n_features = n_features
        self.delay_taps = delay_taps
        self.n_outputs = n_outputs

        input_dim = n_features * (delay_taps + 1)

        layers = []
        prev = input_dim
        for i, h in enumerate(hidden_sizes):
            layers.append(nn.Linear(prev, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            if i < len(hidden_sizes) - 1:
                layers.append(nn.Dropout(dropout))
            prev = h

        layers.append(nn.Linear(prev, n_outputs))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
