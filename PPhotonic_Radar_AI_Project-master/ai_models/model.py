"""
Cognitive Radar AI Module
=======================

This module defines the Deep Learning architecture and inference pipeline.
It implements a Dual-Stream CNN that fuses:
1. Spatial Features (Range-Doppler Map)
2. Temporal Features (Spectrogram)

It provides strict separation between the Model definition and the Inference logic.

Author: Senior AI/ML Engineer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

# --- 1. Model Definition ---

class RadarCNNBranch(nn.Module):
    """Generic CNN Branch for 2D Radar Images."""
    def __init__(self, input_channels: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        
        # Global Average Pooling to handle variable input sizes somewhat, 
        # or fixed size. We assume fixed 128x128 input.
        # 128 -> 64 -> 32 -> 16.
        # 64 channels * 16 * 16 = 16384 features.
        
    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        return x

class DualStreamRadarNet(nn.Module):
    """
    Fused Architecture for Micro-Doppler Classification.
    Inputs:
        x_rd: Range-Doppler Map (Batch, 1, H, W)
        x_spec: Spectrogram (Batch, 1, H, W)
    """
    def __init__(self, num_classes: int = 4):
        super().__init__()
        self.branch_rd = RadarCNNBranch()
        self.branch_spec = RadarCNNBranch()
        
        # Fusion
        # Flat features from one branch: 64 * 16 * 16 = 16384
        # Total flat = 32768
        self.fusion_fc = nn.Linear(32768, 128)
        self.dropout = nn.Dropout(0.5)
        self.head = nn.Linear(128, num_classes)
        
        # XAI Hooks
        self.gradients = None
        
    def forward(self, x_rd, x_spec):
        # Feature Extraction
        f_rd = self.branch_rd(x_rd)
        f_spec = self.branch_spec(x_spec)
        
        # Flatten
        f_rd_flat = f_rd.view(f_rd.size(0), -1)
        f_spec_flat = f_spec.view(f_spec.size(0), -1)
        
        # Fusion
        combined = torch.cat((f_rd_flat, f_spec_flat), dim=1)
        
        x = F.relu(self.fusion_fc(combined))
        x = self.dropout(x)
        logits = self.head(x)
        
        return logits

def list_classes() -> List[str]:
    return ["Drone", "Bird", "Aircraft", "Noise"]


# --- 2. Inference Pipeline ---

@dataclass
class Prediction:
    predicted_class: str
    confidence: float
    probabilities: Dict[str, float]
    attention_map: Optional[np.ndarray] = None # For XAI

class ClassifierPipeline:
    """
    Handles preprocessing, inference, and post-processing.
    Keeps valid state (model weights) separate from execution.
    """
    def __init__(self, model_path: Optional[str] = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.classes = list_classes()
        self.model = DualStreamRadarNet(num_classes=len(self.classes))
        self.model.to(self.device)
        self.model.eval()
        
        if model_path:
            # Load weights if available
            try:
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            except FileNotFoundError:
                print(f"Warning: Model file {model_path} not found. Using Random Weights.")

    def preprocess(self, img_array: np.ndarray) -> torch.Tensor:
        """
        Converts numpy array (H, W) to Tensor (1, 1, 128, 128).
        Resize/Normalize logic goes here.
        """
        # 1. Normalize (Min-Max or Standardization)
        # Assuming input is log-magnitude dB.
        # Robust scalar: (x - min) / (max - min)
        mn = np.min(img_array)
        mx = np.max(img_array)
        if mx - mn > 1e-6:
            norm = (img_array - mn) / (mx - mn)
        else:
            norm = img_array
            
        # 2. To Tensor
        tensor = torch.tensor(norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        
        # 3. Resize (Bilinear)
        tensor = F.interpolate(tensor, size=(128, 128), mode='bilinear', align_corners=False)
        
        return tensor.to(self.device)

    def predict(self, rd_map: np.ndarray, spectrogram: np.ndarray) -> Prediction:
        """
        Runs inference on a single frame.
        """
        t_rd = self.preprocess(rd_map)
        t_spec = self.preprocess(spectrogram)
        
        with torch.no_grad():
            logits = self.model(t_rd, t_spec)
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]
            
        top_idx = np.argmax(probs)
        prob_dict = {cls: float(p) for cls, p in zip(self.classes, probs)}
        
        return Prediction(
            predicted_class=self.classes[top_idx],
            confidence=float(probs[top_idx]),
            probabilities=prob_dict,
            attention_map=None # Hook XAI here later
        )

    # --- Future Upgrade Paths ---
    def upgrade_to_transformer(self):
        """
        [HOOK] Placeholder for Vision Transformer (ViT) backbone.
        Suggested: Replace RadarCNNBranch with a Multi-Head Attention block.
        """
        pass

    def enable_reinforcement_learning(self):
        """
        [HOOK] Placeholder for Cognitive Feedback loop.
        Integrate with an RL Agent to adaptively change Radar parameters (BW, fc)
        based on classification confidence scores.
        """
        pass

if __name__ == "__main__":
    # Smoke Test
    pipeline = ClassifierPipeline()
    dummy_rd = np.random.randn(200, 64) # Random sizes
    dummy_spec = np.random.randn(256, 128)
    
    pred = pipeline.predict(dummy_rd, dummy_spec)
    print(f"Prediction: {pred.predicted_class} ({pred.confidence:.2%})")
    print(f"Probs: {pred.probabilities}")
