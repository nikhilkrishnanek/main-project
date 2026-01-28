"""
Batch Intelligence Engine: High-Throughput Radar Inference
==========================================================

This module provides high-throughput inference capabilities for processing 
large radar datasets or real-time batch streams. It utilizes the 
TacticalHybridClassifier for multimodal intelligence extraction.

Author: Senior AI Performance Engineer
"""

import torch
import torch.nn.functional as F
import numpy as np
import os
from ai_models.architectures import TacticalHybridClassifier, initialize_tactical_model
from ai_models.model import get_tactical_classes


class BatchIntelligenceEngine:
    """
    Performance-optimized engine for batch radar inference.
    """
    def __init__(self, model_checkpoint_path: str = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.target_labels = get_tactical_classes()
        self.num_classes = len(self.target_labels)
        
        # Initialize standardized architecture
        self.model = initialize_tactical_model(num_target_classes=self.num_classes)
        self.model.to(self.device)
        self.model.eval()
        
        if model_checkpoint_path and os.path.exists(model_checkpoint_path):
            try:
                self.model.load_state_dict(torch.load(model_checkpoint_path, map_location=self.device))
                print(f"[INTEL-BATCH] Successfully loaded checkpoint: {model_checkpoint_path}")
            except Exception as e:
                print(f"[INTEL-BATCH] Critical failure during checkpoint loading: {e}")
        else:
            print("[INTEL-BATCH] Proceeding with heuristic initialization.")

    def run_batch_inference(self, 
                            range_doppler_batch: np.ndarray, 
                            kinematic_series_batch: np.ndarray) -> np.ndarray:
        """
        Executes inference on a batch of radar samples.
        
        Args:
            range_doppler_batch: (Batch, H, W) numpy array.
            kinematic_series_batch: (Batch, Seq_Len) numpy array.
            
        Returns:
            Numpy array of shape (Batch, Num_Classes) containing soft probabilities.
        """
        # 1. Tensor Conversion & Preprocessing
        rd_tensor = torch.tensor(range_doppler_batch, dtype=torch.float32).to(self.device)
        ts_tensor = torch.tensor(kinematic_series_batch, dtype=torch.float32).to(self.device)
        
        # 2. Add Channel Dimension if missing (Batch, 1, H, W)
        if rd_tensor.dim() == 3:
            rd_tensor = rd_tensor.unsqueeze(1)
        
        # 3. Spatial Resampling (Standardize to 128x128)
        rd_tensor = F.interpolate(rd_tensor, size=(128, 128), mode='bilinear', align_corners=False)
        
        # 4. Multimodal Inference
        with torch.no_grad():
            logits, _ = self.model(rd_tensor, ts_tensor)
            probabilities = F.softmax(logits, dim=1)
            
        return probabilities.cpu().numpy()

    def get_canonical_labels(self) -> list:
        return self.target_labels
