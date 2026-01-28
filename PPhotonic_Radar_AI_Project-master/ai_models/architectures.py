"""
Tactical Radar Intelligence: Hybrid Neural Architectures
========================================================

This module implements high-performance deep learning architectures tailored 
for photonic radar signatrue analysis. It utilizes a hybrid approach:
1. Spatial-Spectral Encoder (CNN): Processes 2D Range-Doppler maps to extract 
   spatial scattering characteristics.
2. Kinematic-Temporal Encoder (LSTM): Processes 1D Doppler time-series to capture 
   dynamic micro-motion signatures.
3. Feature Fusion Layer: Aggregates both modalities for robust classification.

Architecture Design:
--------------------
- Residual Connections: Mitigates vanishing gradient issues and preserves low-level 
  spectral features.
- Bahdanau Attention: Dynamically weights temporal segments of the Doppler signal 
  to focus on transitionary kinematic events.

Author: Senior AI Research Scientist (Radar Intelligence)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TacticalResidualBlock(nn.Module):
    """
    Implements a 2D Residual block for deep spectral feature extraction.
    """
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super(TacticalResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.identity_mapping = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.identity_mapping = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.identity_mapping(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)


class KinematicAttentionBlock(nn.Module):
    """
    Computes attention weights over temporal kinematic sequences.
    """
    def __init__(self, hidden_dimension: int):
        super(KinematicAttentionBlock, self).__init__()
        self.attention_score_layer = nn.Linear(hidden_dimension, 1)

    def forward(self, sequential_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # sequential_output: (batch_size, sequence_length, hidden_dimension)
        scores = self.attention_score_layer(sequential_output)
        weights = F.softmax(scores, dim=1)
        # Context vector via weighted summation
        context_vector = torch.sum(weights * sequential_output, dim=1)
        return context_vector, weights


class SpatialSpectralEncoder(nn.Module):
    """
    CNN-based encoder for Range-Doppler intensity maps.
    """
    def __init__(self, input_channels: int = 1):
        super(SpatialSpectralEncoder, self).__init__()
        self.stage1 = TacticalResidualBlock(input_channels, 16)
        self.stage2 = TacticalResidualBlock(16, 32, stride=2)
        self.stage3 = TacticalResidualBlock(32, 64, stride=2)
        self.global_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.projection = nn.Linear(64 * 4 * 4, 256)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        return F.relu(self.projection(x))


class KinematicTemporalEncoder(nn.Module):
    """
    RNN-based encoder for 1D Doppler/Kinematic time-series.
    """
    def __init__(self, input_dim: int = 1, hidden_dim: int = 64, num_layers: int = 2):
        super(KinematicTemporalEncoder, self).__init__()
        # Bidirectional LSTM captures future/past context in the Doppler shift
        self.recurrent_unit = nn.LSTM(input_dim, hidden_dim, num_layers, 
                                     batch_first=True, bidirectional=True)
        self.attention_mechanism = KinematicAttentionBlock(hidden_dim * 2)
        self.projection = nn.Linear(hidden_dim * 2, 128)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        sequence_data, _ = self.recurrent_unit(x)
        context_vector, attention_weights = self.attention_mechanism(sequence_data)
        return F.relu(self.projection(context_vector)), attention_weights


class TacticalHybridClassifier(nn.Module):
    """
    State-of-the-art hybrid architecture for radar target classification.
    """
    def __init__(self, num_classes: int = 5):
        super(TacticalHybridClassifier, self).__init__()
        self.spectral_encoder = SpatialSpectralEncoder()
        self.temporal_encoder = KinematicTemporalEncoder()
        
        self.fusion_head = nn.Sequential(
            nn.Linear(256 + 128, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, 
                range_doppler_map: torch.Tensor, 
                kinematic_timeseries: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Multimodal forward pass.
        Args:
            range_doppler_map: (Batch, 1, H, W) tensor.
            kinematic_timeseries: (Batch, Seq_Len) or (Batch, Seq_Len, 1) tensor.
        """
        spatial_features = self.spectral_encoder(range_doppler_map)
        temporal_features, attention_weights = self.temporal_encoder(kinematic_timeseries)
        
        multimodal_concatenation = torch.cat((spatial_features, temporal_features), dim=1)
        classification_logits = self.fusion_head(multimodal_concatenation)
        
        return classification_logits, attention_weights


def initialize_tactical_model(num_target_classes: int = 5) -> TacticalHybridClassifier:
    """Factory function for model instantiation."""
    return TacticalHybridClassifier(num_classes=num_target_classes)
