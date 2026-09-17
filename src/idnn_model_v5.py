"""
Enhanced IDNN v5 -- Physics-Informed, Idle-Normalized Input Delay Neural Network.

Architecture:
  - Input: flattened delay window of 18 physics-informed, idle-normalized features
    (20 taps x 18 features = 378 inputs):
      * Linear acceleration (3)
      * Dynamic gravity vector (3)
      * Gyroscope angular rates (3)
      * Linear acceleration jerk (3)
      * Gyroscope angular jerk (3)
      * Horizontal acceleration norm (decoupled from gravity) (1)
      * Dynamic pitch / suspension tilt angle proxy (1)
      * Idle-normalized road vibration energy (1)
  - Hidden layers: (256, 128, 64) with BatchNorm1d, GELU activations, and Dropout
  - Residual Highway connections for rapid gradient propagation and stable speed estimation
  - Output: 1D vehicle speed (m/s)
"""

import torch
import torch.nn as nn


class IDNNv5(nn.Module):
    def __init__(
        self,
        n_features: int = 18,
        delay_taps: int = 20,
        n_outputs: int = 1,
        hidden_sizes: tuple = (256, 128, 64),
        dropout: float = 0.2,
    ):
        super().__init__()
        self.n_features = n_features
        self.delay_taps = delay_taps
        self.n_outputs = n_outputs

        input_dim = n_features * (delay_taps + 1)

        self.in_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_sizes[0]),
            nn.BatchNorm1d(hidden_sizes[0]),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.block1 = nn.Sequential(
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.BatchNorm1d(hidden_sizes[1]),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.block2 = nn.Sequential(
            nn.Linear(hidden_sizes[1], hidden_sizes[2]),
            nn.BatchNorm1d(hidden_sizes[2]),
            nn.GELU(),
        )

        self.head = nn.Linear(hidden_sizes[2], n_outputs)

    def forward(self, x):
        h0 = self.in_proj(x)
        h1 = self.block1(h0)
        h2 = self.block2(h1)
        out = self.head(h2)
        return out

