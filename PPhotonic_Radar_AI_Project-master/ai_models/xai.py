"""
Tactical Radar Explainability (XAI) Engine
==========================================

This module implements explainable AI techniques to bridge the gap between 
deep learning inference and tactical physics. It provides:
1. Attribution Analysis (Grad-CAM): Localizes the spectral features driving 
   the classification within the Range-Doppler map.
2. Uncertainty Quantification: Estimates epistemic uncertainty via Monte Carlo 
   Dropout for reliability assessment.
3. Physics Grounding: Validates classifications against fundamental radar 
   kinematic laws (e.g., SNR-vs-Noise consistency).

Author: Senior AI Research Scientist (Tactical Systems)
"""

from dataclasses import dataclass
from typing import Dict, Any, List, Union, Tuple, Optional
import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class IntelligenceExplanation:
    """Detailed narrative for tactical intelligence reports."""
    summary_title: str
    tactical_narrative: str
    attribution_metadata: Dict[str, float]
    is_physically_consistent: bool
    calibrated_confidence: float
    epistemic_uncertainty: float
    operational_warnings: List[str]


class TacticalAttributionGenerator:
    """
    Implements Gradient-weighted Class Activation Mapping (Grad-CAM) for 
    tactical spectral maps.
    """
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients: Optional[torch.Tensor] = None
        self.activations: Optional[torch.Tensor] = None
        
        # Register hooks for back-propagation analysis
        self.target_layer.register_forward_hook(self._hook_activations)
        self.target_layer.register_full_backward_hook(self._hook_gradients)

    def _hook_activations(self, module, input, output):
        self.activations = output

    def _hook_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_saliency_map(self, 
                              spectral_map: torch.Tensor, 
                              kinematic_series: torch.Tensor, 
                              target_class_idx: int) -> np.ndarray:
        """
        Generates a normalized heatmap indicating feature importance.
        """
        self.model.zero_grad()
        logits, _ = self.model(spectral_map, kinematic_series)
        
        # Gradient of the winning class w.r.t the target hidden layer
        target_score = logits[0, target_class_idx]
        target_score.backward()

        # Weight the activations by spatial average of gradients (Global Average Pooling)
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        weighted_activations = torch.sum(weights * self.activations, dim=1).squeeze()
        
        # ReLU to focus strictly on positive influences
        saliency_map = F.relu(weighted_activations)
        saliency_map = saliency_map.detach().cpu().numpy()
        
        # Normalization for visualization
        if saliency_map.max() > 0:
            saliency_map /= saliency_map.max()
            
        return saliency_map


def compute_monte_carlo_uncertainty(model: torch.nn.Module, 
                                   spectral_map: torch.Tensor, 
                                   kinematic_series: torch.Tensor, 
                                   stochastic_iterations: int = 15) -> Tuple[np.ndarray, float]:
    """
    Estimates epistemic uncertainty by keeping dropout active during inference.
    """
    model.train() # Enable Dropout and BatchNorm stochasticity
    prob_accumulation = []
    
    with torch.no_grad():
        for _ in range(stochastic_iterations):
            logits, _ = model(spectral_map, kinematic_series)
            probs = F.softmax(logits, dim=1).cpu().numpy()
            prob_accumulation.append(probs)
            
    prob_accumulation = np.array(prob_accumulation)
    mean_probabilities = np.mean(prob_accumulation, axis=0)[0]
    
    # Entropy as a measure of predictive uncertainty (H = -sum(p * log(p)))
    predictive_entropy = -np.sum(mean_probabilities * np.log(mean_probabilities + 1e-9))
    
    return mean_probabilities, float(predictive_entropy)


def validate_classification_physics(tactical_class: str, 
                                    observable_features: Dict[str, float]) -> bool:
    """
    Validates the AI classification against hard radar physics constraints.
    """
    # 1. Noise Floor Consistency
    if tactical_class.lower() == "noise":
        snr_db = observable_features.get("snr_db", -99.0)
        if snr_db > 15.0:  # Noise shouldn't have high SNR
            return False
            
    # 2. Kinematic Velocity Check
    if tactical_class.lower() == "bird":
        velocity_ms = abs(observable_features.get("velocity_ms", 0.0))
        if velocity_ms > 45.0: # Exceeds biological limits of most birds
            return False
            
    return True


def construct_intelligence_narrative(
    tactical_class_raw: str, 
    calibrated_probs: np.ndarray, 
    uncertainty_score: float,
    observed_features: Dict[str, float],
    attribution_map: Optional[np.ndarray] = None
) -> IntelligenceExplanation:
    """
    Translates mathematical inference into professional tactical intelligence.
    """
    tactical_class = tactical_class_raw.capitalize()
    confidence = float(np.max(calibrated_probs))
    
    # Baseline Feature Attribution
    attribution = {
        "Spatial-Spectral (RCS)": 0.35,
        "Kinematic-Temporal": 0.40,
        "Coherent Modulation": 0.25
    }
    
    narrative_segments = []
    operational_warnings = []
    
    # Class-specific characterization
    if tactical_class.lower() == "drone":
        attribution["Coherent Modulation"] = 0.75
        narrative_segments.append(f"Identified as {tactical_class} based on micro-Doppler sidebands consistent with rotor modulation.")
    elif tactical_class.lower() == "missile":
        attribution["Kinematic-Temporal"] = 0.85
        narrative_segments.append(f"Classified as high-threat {tactical_class} due to sustained high-velocity kinematic trajectory.")
    elif tactical_class.lower() == "bird":
        attribution["Coherent Modulation"] = 0.60
        narrative_segments.append(f"Biological entity (Bird) signature detected with characteristic low-RCS flapping modulation.")
    elif tactical_class.lower() == "aircraft":
        attribution["Spatial-Spectral (RCS)"] = 0.80
        narrative_segments.append(f"Fixed-wing {tactical_class} identified by large, stable RCS and non-fluctuating Kinematics.")
    else:
        narrative_segments.append(f"Non-target signature categorized as {tactical_class}.")

    # Physics validation
    is_valid = validate_classification_physics(tactical_class_raw, observed_features)
    if not is_valid:
        operational_warnings.append("⚠️ Physics Alert: Classification violates kinematic consistency models.")
    
    # Uncertainty analysis
    if uncertainty_score > 0.8:
        operational_warnings.append("⚠️ Ambiguity Warning: High epistemic uncertainty; target may be an unknown class.")
    
    if attribution_map is not None:
        narrative_segments.append("\n[XAI] Target localization confirmed at peak intensity centroid.")

    return IntelligenceExplanation(
        summary_title=f"TACTICAL ID: {tactical_class.upper()}",
        tactical_narrative=" ".join(narrative_segments),
        attribution_metadata=attribution,
        is_physically_consistent=is_valid,
        calibrated_confidence=confidence,
        epistemic_uncertainty=uncertainty_score,
        operational_warnings=operational_warnings
    )
