import torch
import torch.nn as nn

class LightweightIDNN(nn.Module):
    def __init__(self, window_size=20, features=6, hidden_units=32):
        super(LightweightIDNN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(window_size * features, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, 2)
        )

    def forward(self, x):
        flat_input = x.view(x.size(0), -1)
        return self.network(flat_input)