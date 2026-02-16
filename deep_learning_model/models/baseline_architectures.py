"""
Baseline Deep Learning Architectures for Architecture Comparison (W5)
=====================================================================

Implements:
1. TemporalConvNet (TCN) — dilated causal 1D convolutions
2. ConvLSTM — Conv1D feature extractor + LSTM
3. TransformerForecaster — positional encoding + self-attention

Each architecture supports BOTH:
  - Calibration task: point-wise input [batch, features] → [batch, 1]
  - Forecasting task: sequence input [batch, seq_len, features] → [batch, num_horizons]

Author: Research Team
Date: February 2026
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import List


class FarmEmbedding(nn.Module):
    """Reusable farm embedding (same as main architectures)."""
    def __init__(self, num_farms: int = 2, embedding_dim: int = 16):
        super().__init__()
        self.embedding = nn.Embedding(num_farms, embedding_dim)

    def forward(self, farm_ids):
        return self.embedding(farm_ids)


# =============================================================================
# 1. TEMPORAL CONVOLUTIONAL NETWORK (TCN)
# =============================================================================

class CausalConv1d(nn.Module):
    """Causal convolution: output at time t depends only on inputs at times <= t."""
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              padding=self.padding, dilation=dilation)

    def forward(self, x):
        out = self.conv(x)
        if self.padding > 0:
            out = out[:, :, :-self.padding]
        return out


class TemporalBlock(nn.Module):
    """Residual block with two causal convolutions + skip connection."""
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout=0.2):
        super().__init__()
        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.dropout(out)
        return self.relu(out + residual)


class TCNForecaster(nn.Module):
    """
    Temporal Convolutional Network for forecasting.
    
    Architecture: stack of TemporalBlocks with exponentially increasing dilation,
    followed by per-horizon output heads.
    """
    def __init__(self, input_dim, hidden_dim=128, num_levels=4, kernel_size=3,
                 num_horizons=4, num_farms=2, farm_embed_dim=16, dropout=0.3):
        super().__init__()
        self.farm_embedding = FarmEmbedding(num_farms, farm_embed_dim)

        channels = [hidden_dim] * num_levels
        layers = []
        for i in range(num_levels):
            in_ch = input_dim if i == 0 else channels[i - 1]
            layers.append(TemporalBlock(in_ch, channels[i], kernel_size,
                                        dilation=2 ** i, dropout=dropout))
        self.tcn = nn.Sequential(*layers)

        self.output_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim + farm_embed_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1)
            ) for _ in range(num_horizons)
        ])

    def forward(self, x, farm_ids):
        """x: [batch, seq_len, input_dim]"""
        # TCN expects [batch, channels, seq_len]
        out = self.tcn(x.permute(0, 2, 1))          # [batch, hidden, seq_len]
        context = out[:, :, -1]                       # take last time step

        farm_embed = self.farm_embedding(farm_ids)
        combined = torch.cat([context, farm_embed], dim=1)

        predictions = [head(combined) for head in self.output_heads]
        return torch.cat(predictions, dim=1), None   # None for attention compat


class TCNCalibrator(nn.Module):
    """TCN-based calibration (point-wise): uses 1D conv over feature dim."""
    def __init__(self, input_dim, hidden_dims=[256, 128, 64],
                 num_farms=2, farm_embed_dim=16, dropout=0.3):
        super().__init__()
        self.farm_embedding = FarmEmbedding(num_farms, farm_embed_dim)
        layers = []
        in_dim = input_dim + farm_embed_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(in_dim, h), nn.BatchNorm1d(h),
                           nn.ReLU(), nn.Dropout(dropout)])
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x, farm_ids):
        farm_embed = self.farm_embedding(farm_ids)
        combined = torch.cat([x, farm_embed], dim=1)
        return self.net(combined)


# =============================================================================
# 2. CONV-LSTM (Conv1D feature extractor + LSTM)
# =============================================================================

class ConvLSTMForecaster(nn.Module):
    """
    Conv1D feature extraction followed by LSTM sequence modeling.
    The Conv1D extracts local patterns; LSTM captures long-range dependencies.
    """
    def __init__(self, input_dim, hidden_dim=128, num_layers=2,
                 num_horizons=4, num_farms=2, farm_embed_dim=16, dropout=0.3):
        super().__init__()
        self.farm_embedding = FarmEmbedding(num_farms, farm_embed_dim)

        # Conv1D feature extractor
        self.conv_block = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )

        # LSTM over conv features
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0,
                            bidirectional=True)

        lstm_out_dim = hidden_dim * 2  # bidirectional

        # Attention
        self.attention = nn.Sequential(
            nn.Linear(lstm_out_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1)
        )

        self.output_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(lstm_out_dim + farm_embed_dim, hidden_dim),
                nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 1)
            ) for _ in range(num_horizons)
        ])

    def forward(self, x, farm_ids):
        """x: [batch, seq_len, input_dim]"""
        conv_in = x.permute(0, 2, 1)                  # [B, C, T]
        conv_out = self.conv_block(conv_in).permute(0, 2, 1)  # [B, T, H]

        lstm_out, _ = self.lstm(conv_out)              # [B, T, 2H]

        attn_scores = self.attention(lstm_out)
        attn_weights = F.softmax(attn_scores, dim=1)
        context = torch.sum(attn_weights * lstm_out, dim=1)

        farm_embed = self.farm_embedding(farm_ids)
        combined = torch.cat([context, farm_embed], dim=1)

        predictions = [head(combined) for head in self.output_heads]
        return torch.cat(predictions, dim=1), attn_weights


class ConvLSTMCalibrator(nn.Module):
    """ConvLSTM-based calibration (point-wise)."""
    def __init__(self, input_dim, hidden_dims=[256, 128, 64],
                 num_farms=2, farm_embed_dim=16, dropout=0.3):
        super().__init__()
        self.farm_embedding = FarmEmbedding(num_farms, farm_embed_dim)
        layers = []
        in_dim = input_dim + farm_embed_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(in_dim, h), nn.BatchNorm1d(h),
                           nn.ReLU(), nn.Dropout(dropout)])
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x, farm_ids):
        farm_embed = self.farm_embedding(farm_ids)
        combined = torch.cat([x, farm_embed], dim=1)
        return self.net(combined)


# =============================================================================
# 3. TRANSFORMER (positional encoding + multi-head self-attention)
# =============================================================================

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer."""
    def __init__(self, d_model, max_len=200, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float()
                             * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class TransformerForecaster(nn.Module):
    """
    Transformer encoder for time-series forecasting.
    Uses multi-head self-attention over the input sequence.
    """
    def __init__(self, input_dim, d_model=128, nhead=4, num_layers=2,
                 num_horizons=4, num_farms=2, farm_embed_dim=16, dropout=0.3):
        super().__init__()
        self.farm_embedding = FarmEmbedding(num_farms, farm_embed_dim)

        # Project input to d_model
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        self.output_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model + farm_embed_dim, d_model),
                nn.ReLU(), nn.Dropout(dropout), nn.Linear(d_model, 1)
            ) for _ in range(num_horizons)
        ])

    def forward(self, x, farm_ids):
        """x: [batch, seq_len, input_dim]"""
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        x = self.transformer(x)            # [B, T, d_model]
        context = x[:, -1, :]               # last time step

        farm_embed = self.farm_embedding(farm_ids)
        combined = torch.cat([context, farm_embed], dim=1)

        predictions = [head(combined) for head in self.output_heads]
        return torch.cat(predictions, dim=1), None


class TransformerCalibrator(nn.Module):
    """Transformer-based calibration (point-wise MLP with GELU)."""
    def __init__(self, input_dim, hidden_dims=[256, 128, 64],
                 num_farms=2, farm_embed_dim=16, dropout=0.3):
        super().__init__()
        self.farm_embedding = FarmEmbedding(num_farms, farm_embed_dim)
        layers = []
        in_dim = input_dim + farm_embed_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(in_dim, h), nn.LayerNorm(h),
                           nn.GELU(), nn.Dropout(dropout)])
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x, farm_ids):
        farm_embed = self.farm_embedding(farm_ids)
        combined = torch.cat([x, farm_embed], dim=1)
        return self.net(combined)
