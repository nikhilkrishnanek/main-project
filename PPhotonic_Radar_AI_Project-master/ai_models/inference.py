import torch
import torch.nn.functional as F
import numpy as np
import os
from ai.model import DualStreamRadarNet as build_pytorch_model
from core.config import get_config

class InferenceEngine:
    def __init__(self, model_path=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = get_config()
        self.num_classes = 6 # defined in config usually
        self.metadata_size = 8
        
        self.model = build_pytorch_model(num_classes=self.num_classes)
        self.model.to(self.device)
        self.model.eval()
        
        if model_path and os.path.exists(model_path):
            try:
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                print(f"Loaded model from {model_path}")
            except Exception as e:
                print(f"Failed to load model: {e}")
        else:
            print("Initialized with random weights (no model found)")

    def predict(self, rd_map, spectrogram, metadata):
        """
        Run inference on single sample.
        rd_map: (H, W)
        spectrogram: (H, W) or other
        metadata: (8,)
        """
        # Prepare tensors
        rd_tensor = torch.tensor(rd_map, dtype=torch.float32).to(self.device)
        spec_tensor = torch.tensor(spectrogram, dtype=torch.float32).to(self.device)
        meta_tensor = torch.tensor(metadata, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        # Add batch and channel dims -> (1, 1, H, W)
        rd_tensor = rd_tensor.unsqueeze(0).unsqueeze(0)
        spec_tensor = spec_tensor.unsqueeze(0).unsqueeze(0)
        
        # Resize to expected model input (128, 128)
        rd_tensor = F.interpolate(rd_tensor, size=(128, 128), mode='bilinear', align_corners=False)
        spec_tensor = F.interpolate(spec_tensor, size=(128, 128), mode='bilinear', align_corners=False)
        
        with torch.no_grad():
            logits = self.model(rd_tensor, spec_tensor, meta_tensor)
            probs = torch.nn.functional.softmax(logits, dim=1)
            
        return probs.cpu().numpy()[0] # Return 1D array of probs

    def get_classes(self):
        return ["Drone", "Bird", "Aircraft", "Missile", "Helicopter", "Clutter"]
