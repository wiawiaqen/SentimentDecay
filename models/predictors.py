"""
This module contains implementations for stock prediction models.
"""
import torch
import torch.nn as nn

class StockPredictor(nn.Module):
    def __init__(self, input_dim, output_dim=1):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

    def forward(self, x):
        raise NotImplementedError

class LSTMPredictor(StockPredictor):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2):
        super().__init__(input_dim)
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True, 
            dropout=0.2
        )
        self.linear = nn.Linear(hidden_dim, self.output_dim)

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        lstm_out, _ = self.lstm(x)
        predictions = self.linear(lstm_out[:, -1, :])  # Use last timestep
        return predictions