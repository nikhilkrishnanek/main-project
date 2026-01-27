"""
Hybrid CNN-LSTM Radar Architecture (CRNN)
=========================================

This module defines advanced neural architectures for Radar Intelligence.
1. CNN Branch: Extracts spatial features from 2D Spectrograms.
2. LSTM Branch: Extracts temporal features from 1D Doppler time-series.
3. Hybrid Fusion: Combines both branches for robust threat classification.

Author: Radar AI Engineer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class SpectralCNN(nn.Module):
    """
    CNN Branch for processing 2D Spectrograms.
    """
    def __init__(self, in_channels=1):
        super(SpectralCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)
        # Assuming input 128x128 -> after 2 pools -> 32x32
        self.fc = nn.Linear(32 * 32 * 32, 256) 

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc(x))
        return x

class TemporalLSTM(nn.Module):
    """
    LSTM Branch for processing 1D Doppler time-series.
    """
    def __init__(self, input_size=1, hidden_size=64, num_layers=2):
        super(TemporalLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 128)

    def forward(self, x):
        # x shape: (batch, seq_len) -> (batch, seq_len, 1)
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        _, (h_n, _) = self.lstm(x)
        # Use last hidden state
        out = F.relu(self.fc(h_n[-1]))
        return out

class HybridRadarNet(nn.Module):
    """
    Fused CNN-LSTM Architecture for spatiotemporal target classification.
    """
    def __init__(self, num_classes=4):
        super(HybridRadarNet, self).__init__()
        self.cnn = SpectralCNN()
        self.lstm = TemporalLSTM()
        
        # Fusion Layer
        self.classifier = nn.Sequential(
            nn.Linear(256 + 128, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, spectrogram, time_series):
        cnn_feat = self.cnn(spectrogram)
        lstm_feat = self.lstm(time_series)
        
        # Concatenate features
        fused = torch.cat((cnn_feat, lstm_feat), dim=1)
        logits = self.classifier(fused)
        return logits

def get_hybrid_model(num_classes=4):
    return HybridRadarNet(num_classes=num_classes)

if __name__ == "__main__":
    model = get_hybrid_model()
    spec = torch.randn(8, 1, 128, 128)
    ts = torch.randn(8, 1000)
    out = model(spec, ts)
    print(f"Model output shape: {out.shape} | Classes: {torch.argmax(out, dim=1)}")
