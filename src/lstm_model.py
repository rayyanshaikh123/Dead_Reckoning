"""
LSTM Speed Predictor — Best of v1 + v2 + temporal dynamics.

Hybrid architecture:
  1. LSTM processes per-timestep features (raw sensors + jerk) — captures temporal dynamics
  2. Window statistics (from v2) concatenated with LSTM output — ensures robustness
  3. MLP head produces speed prediction
  4. Kalman filter post-processes for smoothing

Includes jerk (rate of change of acceleration/gyroscope) as novel features.
"""

import numpy as np
import torch
import torch.nn as nn


class LSTMSpeedPredictor(nn.Module):
    """
    Hybrid LSTM model.

    Inputs:
        x_seq:  (batch, seq_len, 20) — per-timestep sensor features through LSTM
        x_stat: (batch, 26) — window statistics concatenated after LSTM

    Output:
        (batch, 1) — predicted vehicle speed (m/s)
    """

    def __init__(
        self,
        n_seq_features: int = 20,
        n_stat_features: int = 26,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.n_seq_features = n_seq_features
        self.n_stat_features = n_stat_features
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM for temporal dynamics
        self.lstm = nn.LSTM(
            input_size=n_seq_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        # MLP head: LSTM last hidden + window statistics
        mlp_in = hidden_size + n_stat_features
        self.head = nn.Sequential(
            nn.Linear(mlp_in, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x_seq, x_stat):
        lstm_out, (hn, cn) = self.lstm(x_seq)
        last_hidden = hn[-1]                          # (batch, hidden)
        combined = torch.cat([last_hidden, x_stat], dim=1)
        return self.head(combined)


# ================================================================
# Kalman Filter
# ================================================================

def kalman_filter_speed(
    raw_speed: np.ndarray,
    process_noise: float = 0.5,
    measurement_noise: float = 4.0,
) -> np.ndarray:
    """
    1-D Kalman filter on predicted speed.

    State model: constant-speed (random walk).
    Measurement: LSTM speed prediction.

    Args:
        raw_speed:        (N,) LSTM raw predictions in m/s
        process_noise:    Q — how much true speed can change per sample
        measurement_noise: R — expected variance of LSTM prediction error

    Returns:
        filtered_speed:   (N,) smoothed, non-negative
    """
    n = len(raw_speed)
    out = np.zeros(n, dtype=np.float64)

    x = float(raw_speed[0])
    P = 1.0

    for i in range(n):
        # Predict
        x_pred = x
        P_pred = P + process_noise

        # Update
        K = P_pred / (P_pred + measurement_noise)
        x = x_pred + K * (raw_speed[i] - x_pred)
        P = (1.0 - K) * P_pred

        out[i] = max(x, 0.0)

    return out

