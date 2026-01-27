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

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)

class AttentionBlock(nn.Module):
    """
    Bahdanau-style Attention for focusing on key temporal events in Doppler series.
    """
    def __init__(self, hidden_size):
        super(AttentionBlock, self).__init__()
        self.attn = nn.Linear(hidden_size, 1)

    def forward(self, lstm_output):
        # lstm_output: (batch, seq_len, hidden_size)
        attn_weights = F.softmax(self.attn(lstm_output), dim=1)
        # weights: (batch, seq_len, 1)
        context = torch.sum(attn_weights * lstm_output, dim=1)
        return context, attn_weights

class SpectralCNN(nn.Module):
    def __init__(self, in_channels=1):
        super(SpectralCNN, self).__init__()
        self.layer1 = ResidualBlock(in_channels, 16)
        self.layer2 = ResidualBlock(16, 32, stride=2)
        self.layer3 = ResidualBlock(32, 64, stride=2)
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc = nn.Linear(64 * 4 * 4, 256)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return F.relu(self.fc(x))

class TemporalLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2):
        super(TemporalLSTM, self).__init__()
        # Use bidirectional for better context
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.attention = AttentionBlock(hidden_size * 2)
        self.fc = nn.Linear(hidden_size * 2, 128)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        out, _ = self.lstm(x)
        context, attn_weights = self.attention(out)
        return F.relu(self.fc(context)), attn_weights

class HybridRadarNet(nn.Module):
    def __init__(self, num_classes=5):
        super(HybridRadarNet, self).__init__()
        self.cnn = SpectralCNN()
        self.lstm = TemporalLSTM()
        
        self.classifier = nn.Sequential(
            nn.Linear(256 + 128, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )

    def forward(self, spectrogram, time_series):
        cnn_feat = self.cnn(spectrogram)
        lstm_feat, attn_weights = self.lstm(time_series)
        
        fused = torch.cat((cnn_feat, lstm_feat), dim=1)
        logits = self.classifier(fused)
        return logits, attn_weights

def get_hybrid_model(num_classes=4):
    return HybridRadarNet(num_classes=num_classes)

if __name__ == "__main__":
    model = get_hybrid_model()
    spec = torch.randn(8, 1, 128, 128)
    ts = torch.randn(8, 1000)
    out = model(spec, ts)
    print(f"Model output shape: {out.shape} | Classes: {torch.argmax(out, dim=1)}")
