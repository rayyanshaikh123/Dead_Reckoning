"""
IDNN v2 Model — Feature-engineered for phone-in-use robustness.

Key differences from v1:
- Uses window statistics (mean, std, median) instead of raw time-series
- Adds spectral band energies from accelerometer FFT
- Includes magnetometer heading (immune to hand motion)
- Uses orientation sensor (low-pass via window mean)
- Gravity stability metric (phone-use detector)
- BatchNorm for handling diverse feature scales
- All features inherently robust to hand motion
"""

import torch
import torch.nn as nn


class IDNNv2(nn.Module):
    """
    Input Delay Neural Network v2.

    Takes pre-extracted window features (statistics + spectral + heading)
    instead of raw time-delayed sensor values.

    Architecture adds BatchNorm for better handling of
    multi-scale features (spectral energies vs. acceleration means).
    """

    def __init__(
        self,
        n_features: int = 33,
        n_outputs: int = 1,
        hidden_sizes: tuple = (256, 128, 64),
        dropout: float = 0.2,
    ):
        super().__init__()

        self.n_features = n_features
        self.n_outputs = n_outputs

        layers = []
        prev_dim = n_features

        for i, h in enumerate(hidden_sizes):
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            if i < len(hidden_sizes) - 1:
                layers.append(nn.Dropout(dropout))
            prev_dim = h

        layers.append(nn.Linear(prev_dim, n_outputs))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """
        Args:
            x: (batch, n_features) — pre-extracted window features

        Returns:
            (batch, n_outputs)
        """
        return self.network(x)

