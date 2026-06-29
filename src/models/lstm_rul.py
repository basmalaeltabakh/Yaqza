"""LSTM RUL model with modern improvements: dropout, bidirectional, attention."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ImprovedLSTM(nn.Module):
    """LSTM with dropout, layer normalization, and optional attention.

    **Architecture**:
    - Stacked LSTM layers (optional bidirectional).
    - Dropout for regularization.
    - Layer normalization on inputs.
    - Global average pooling before fully-connected layer.

    **Hyperparameters**:
    - ``hidden_size``: LSTM hidden dimension.
    - ``num_layers``: Number of LSTM stacks.
    - ``dropout``: Probability of dropping LSTM outputs.
    - ``bidirectional``: Whether to use bidirectional LSTM.
    - ``use_attention``: Whether to add multi-head self-attention.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = False,
        use_attention: bool = False,
        output_size: int = 1,
    ):
        """Initialize LSTM model.

        Args:
            input_size:     Number of input features.
            hidden_size:    LSTM hidden dimension.
            num_layers:     Number of stacked LSTM layers.
            dropout:        Dropout probability.
            bidirectional:  Use bidirectional LSTM.
            use_attention:  Add attention layer.
            output_size:    Number of outputs (default 1 for RUL).
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.use_attention = use_attention

        # Layer normalization for input stability
        self.ln_input = nn.LayerNorm(input_size)

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
            batch_first=True,
        )

        lstm_output_size = hidden_size * (2 if bidirectional else 1)

        # Optional attention (multi-head self-attention on LSTM outputs)
        if use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=lstm_output_size,
                num_heads=4,
                dropout=dropout,
                batch_first=True,
            )

        # Fully connected layers with dropout
        self.fc1 = nn.Linear(lstm_output_size, 64)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, seq_len, input_size).

        Returns:
            RUL predictions of shape (batch, output_size).
        """
        # Layer norm on input
        x = self.ln_input(x)

        # LSTM pass
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, lstm_output_size)

        # Optional attention
        if self.use_attention:
            attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
            lstm_out = lstm_out + attn_out  # Residual connection

        # Global average pooling
        pooled = torch.mean(lstm_out, dim=1)  # (batch, lstm_output_size)

        # FC layers
        out = self.fc1(pooled)
        out = F.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)  # (batch, output_size)

        return out


class LSTMWithAttention(nn.Module):
    """LSTM with scaled dot-product attention on sequence outputs.

    **Use this when**: You want to focus on specific timesteps that matter
    for RUL prediction. Attention learns *which* cycles in the window are
    most predictive of degradation.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        """Initialize LSTM with attention.

        Args:
            input_size: Number of features.
            hidden_size: LSTM dimension.
            num_layers: Stacked LSTM layers.
            dropout: Dropout probability.
        """
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )

        # Attention
        self.attention = nn.Linear(hidden_size, 1)

        # Classifier
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with attention.

        Args:
            x: (batch, seq_len, input_size)

        Returns:
            (batch, 1) – RUL prediction
        """
        lstm_out, (h_n, c_n) = self.lstm(x)  # (batch, seq_len, hidden_size)

        # Attention weights
        attn_weights = self.attention(lstm_out)  # (batch, seq_len, 1)
        attn_weights = F.softmax(attn_weights, dim=1)  # Normalize over time

        # Weighted sum
        context = torch.sum(attn_weights * lstm_out, dim=1)  # (batch, hidden_size)

        # Predict RUL
        out = self.fc(context)
        return out
