"""ConvLSTM temporal trend model for glacial lake expansion classification.

Stage 4 of the SIREN ML pipeline (PRD §9.3, ADR-002). Takes a sequence
of satellite-derived water masks (or change masks) over time and
classifies the temporal trend into:

  0: stable       — no significant change
  1: slowly       — gradual expansion (early warning signal)
  2: rapidly      — rapid expansion (critical alert)
  3: uncertain    — fluctuating/noisy, cannot classify confidently

Architecture:
  - ConvLSTM cells process the spatial-temporal sequence
  - A CNN encoder extracts spatial features from each timestep
  - The final hidden state is classified into 4 trend categories

Design rule (PRD §9.3): deliberately does NOT predict a precise collapse
timestamp. Only classifies progression velocity. This is scientifically
responsible given satellite revisit intervals of 1-6 days.

Training: synthetic sequences derived from Sen1Floods11 chips, where
water area is progressively expanded over N timesteps to simulate
different expansion rates.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvLSTMCell(nn.Module):
    """Convolutional LSTM cell — processes spatial data over time.

    Unlike a standard LSTM, the input/output transformations use
    convolutions instead of linear layers, preserving spatial structure.
    """

    def __init__(self, input_dim: int, hidden_dim: int, kernel_size: int = 3) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        padding = kernel_size // 2

        # Single conv for all 4 gates (input, forget, output, candidate)
        self.conv = nn.Conv2d(
            input_dim + hidden_dim,
            4 * hidden_dim,
            kernel_size=kernel_size,
            padding=padding,
            bias=True,
        )

    def forward(
        self,
        x: torch.Tensor,
        h_prev: torch.Tensor,
        c_prev: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for one timestep.

        Args:
            x: Input at current timestep (B, input_dim, H, W)
            h_prev: Hidden state from previous timestep (B, hidden_dim, H, W)
            c_prev: Cell state from previous timestep (B, hidden_dim, H, W)

        Returns:
            h_new, c_new: Updated hidden and cell states
        """
        combined = torch.cat([x, h_prev], dim=1)
        gates = self.conv(combined)
        i, f, o, g = torch.chunk(gates, 4, dim=1)

        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)

        c_new = f * c_prev + i * g
        h_new = o * torch.tanh(c_new)

        return h_new, c_new


class SpatialEncoder(nn.Module):
    """Lightweight CNN encoder for spatial feature extraction per timestep.

    Takes a single-channel water mask (or multi-channel satellite chip)
    and extracts a low-dimensional spatial feature map.
    """

    def __init__(self, in_channels: int = 1, feature_dim: int = 32) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # /2

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # /4

            nn.Conv2d(32, feature_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True),
        )
        self.feature_dim = feature_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract spatial features from a single timestep.

        Args:
            x: (B, C, H, W) — satellite chip or water mask at one timestep

        Returns:
            (B, feature_dim, H/4, W/4) — spatial feature map
        """
        return self.features(x)


class ConvLSTMTrendClassifier(nn.Module):
    """Full ConvLSTM temporal trend classifier.

    Takes a sequence of water masks (or satellite chips) and classifies
    the temporal trend into: stable, slowly expanding, rapidly expanding,
    or uncertain.

    Input:  (B, T, C, H, W) — T timesteps of C-channel images
    Output: (B, 4) — class logits [stable, slowly, rapidly, uncertain]
    """

    def __init__(
        self,
        in_channels: int = 1,
        encoder_dim: int = 32,
        lstm_dim: int = 64,
        num_classes: int = 4,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.encoder_dim = encoder_dim
        self.lstm_dim = lstm_dim
        self.num_classes = num_classes

        # Spatial encoder (shared across timesteps)
        self.encoder = SpatialEncoder(in_channels=in_channels, feature_dim=encoder_dim)

        # ConvLSTM cell
        self.lstm_cell = ConvLSTMCell(
            input_dim=encoder_dim,
            hidden_dim=lstm_dim,
            kernel_size=3,
        )

        # Classifier head — takes the final hidden state's global average pool
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(lstm_dim, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(32, num_classes),
        )

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        """Process a temporal sequence and output trend classification.

        Args:
            sequence: (B, T, C, H, W) — batch of T-timestep sequences

        Returns:
            (B, num_classes) — class logits
        """
        B, T, C, H, W = sequence.shape

        # Initialize LSTM states
        # Compute spatial dims after encoder (2 maxpools = /4)
        enc_h, enc_w = H // 4, W // 4
        h = torch.zeros(B, self.lstm_dim, enc_h, enc_w, device=sequence.device)
        c = torch.zeros(B, self.lstm_dim, enc_h, enc_w, device=sequence.device)

        # Process each timestep
        for t in range(T):
            x_t = sequence[:, t]  # (B, C, H, W)
            feat_t = self.encoder(x_t)  # (B, encoder_dim, H/4, W/4)
            h, c = self.lstm_cell(feat_t, h, c)

        # Classify the final hidden state
        logits = self.classifier(h)
        return logits

    def predict_trend(self, sequence: torch.Tensor) -> tuple[int, float]:
        """Convenience method for inference — returns class index + confidence.

        Args:
            sequence: (1, T, C, H, W) — single sequence

        Returns:
            (class_index, confidence) — predicted trend class and softmax confidence
        """
        self.eval()
        with torch.no_grad():
            logits = self(sequence)
            probs = F.softmax(logits, dim=1)
            conf, pred = probs.max(dim=1)
        return pred.item(), conf.item()


# Class names for interpretation
TREND_CLASSES = ["stable", "slowly", "rapidly", "uncertain"]
TREND_CLASS_TO_IDX = {name: idx for idx, name in enumerate(TREND_CLASSES)}
