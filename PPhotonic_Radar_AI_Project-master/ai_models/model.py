"""
Radar Intelligence Pipeline: Target Classification & Characterization
=====================================================================

This module provides the high-level interface for AI-driven target 
recognition. It encapsulates the deep neural model, preprocessing logic, 
and post-inference probability calibration.

The pipeline is designed for real-time tactical environments, strictly 
separating model weights and architecture from the execution context.

Intelligence Output:
--------------------
- Predicted Class Label: Canonical target type (e.g., Drone, Missile).
- Inference Confidence: Statistical certainty of the classification.
- Probability Distribution: Full soft-max output for cognitive feedback.

Author: Lead AI/ML Radar Architect
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
from ai_models.architectures import initialize_tactical_model


@dataclass
class IntelligenceOutput:
    """Standardized vessel/target intelligence report."""
    tactical_class: str
    inference_confidence: float
    class_probabilities: Dict[str, float]
    attention_weights: Optional[np.ndarray] = None


def get_tactical_classes() -> List[str]:
    """Returns the set of targets the intelligence model is trained to recognize."""
    return ["Drone", "Bird", "Aircraft", "Missile", "Noise"]


class IntelligencePipeline:
    """
    Orchestration layer for Radar Artificial Intelligence.
    Handles data normalization, tensor conversion, and model execution.
    """
    def __init__(self, model_checkpoint_path: Optional[str] = None):
        """
        Initializes the intelligence pipeline.
        
        Args:
            model_checkpoint_path: Optional path to serialized PyTorch weights.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.target_labels = get_tactical_classes()
        
        # Instantiate standardized tactical classifier
        self.model = initialize_tactical_model(num_target_classes=len(self.target_labels))
        self.model.to(self.device)
        self.model.eval()
        
        if model_checkpoint_path:
            try:
                self.model.load_state_dict(torch.load(model_checkpoint_path, map_location=self.device))
            except FileNotFoundError:
                print(f"[RECON-AI] Warning: Specified checkpoint not found. Operating with heuristic weights.")

    def preprocess_spectral_map(self, rd_intensity_map: np.ndarray) -> torch.Tensor:
        """
        Prepares 2D Range-Doppler maps for CNN ingestion.
        """
        # 1. Normalization (Min-Max Scaling to [0, 1])
        # Assumes input in dB or linear power
        val_min = np.min(rd_intensity_map)
        val_max = np.max(rd_intensity_map)
        
        if val_max - val_min > 1e-9:
            normalized_map = (rd_intensity_map - val_min) / (val_max - val_min)
        else:
            normalized_map = rd_intensity_map
            
        # 2. Tensor Conversion & Dimensionality Alignment
        # (H, W) -> (Batch=1, Channels=1, H, W)
        tensor = torch.tensor(normalized_map, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        
        # 3. Spatial Resampling (Bilinear Interpolation)
        # Standardizes input size to model's receptive field
        resampled_tensor = F.interpolate(tensor, size=(128, 128), mode='bilinear', align_corners=False)
        
        return resampled_tensor.to(self.device)

    def preprocess_kinematic_stream(self, doppler_time_series: np.ndarray) -> torch.Tensor:
        """
        Prepares 1D Doppler/Kinematic sequences for RNN ingestion.
        """
        # Ensure sequence is a float tensor with batch dimension
        tensor = torch.tensor(doppler_time_series, dtype=torch.float32)
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(0) # Add batch dim
            
        return tensor.to(self.device)

    def infer_tactical_intelligence(self, 
                                   rd_intensity_map: np.ndarray, 
                                   doppler_time_series: np.ndarray) -> IntelligenceOutput:
        """
        Executes multimodal inference to classify a radar tactical entity.
        """
        # Pre-processing
        tensor_map = self.preprocess_spectral_map(rd_intensity_map)
        tensor_stream = self.preprocess_kinematic_stream(doppler_time_series)
        
        with torch.no_grad():
            # Dual-path forward pass
            logits, attention_weights = self.model(tensor_map, tensor_stream)
            
            # Probability calibration
            soft_probabilities = F.softmax(logits, dim=1).cpu().numpy()[0]
            
        prediction_idx = np.argmax(soft_probabilities)
        probability_map = {label: float(p) for label, p in zip(self.target_labels, soft_probabilities)}
        
        return IntelligenceOutput(
            tactical_class=self.target_labels[prediction_idx],
            inference_confidence=float(soft_probabilities[prediction_idx]),
            class_probabilities=probability_map,
            attention_weights=attention_weights.cpu().numpy() if attention_weights is not None else None
        )
