"""
Tactical Intelligence Training Entry Point
==========================================

This script serves as the primary entry point for training the 
TacticalHybridClassifier. It orchestrates high-fidelity dataset synthesis, 
multimodal model initialization, and standardized persistence of the 
resulting intelligence weights.

Author: Senior AI Research Scientist (Radar Intelligence)
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from ai_models.dataset_generator import RadarDatasetGenerator
from ai_models.architectures import initialize_tactical_model
from ai_models.model import get_tactical_classes


def generate_tactical_training_data(samples_per_vessel_class: int = 50):
    """
    Orchestrates the synthesis of a multimodal training dataset.
    """
    # High-resolution configuration for Micro-Doppler signature extraction
    config = {"chirp_duration_s": 0.1, "sampling_rate_hz": 5e5}
    data_generator = RadarDatasetGenerator(config)
    
    print("[TRAIN-IO] Synthesizing high-fidelity multimodal radar dataset...")
    multimodal_batch = data_generator.generate_batch(samples_per_class=samples_per_vessel_class)
    
    return (
        multimodal_batch['rd_maps'],
        multimodal_batch['spectrograms'],
        multimodal_batch['time_series'],
        multimodal_batch['labels']
    )


def execute_intelligence_training_entrypoint(num_epochs: int = 5):
    """
    Main training workflow for the radar intelligence model.
    """
    # 1. Dataset Acquisition
    _, spectral_maps, kinematic_series, ground_truth = generate_tactical_training_data(samples_per_vessel_class=25)
    
    # 2. Data Loading Infrastructure
    training_dataset = TensorDataset(spectral_maps, kinematic_series, ground_truth)
    intel_loader = DataLoader(training_dataset, batch_size=16, shuffle=True)

    # 3. Model & Optimization Orchestration
    target_classes = get_tactical_classes()
    intelligence_model = initialize_tactical_model(num_target_classes=len(target_classes))
    
    optimizer = optim.Adam(intelligence_model.parameters(), lr=0.001)
    objective_function = nn.CrossEntropyLoss()

    print(f"[TRAIN-IO] Starting training cycle for {len(target_classes)} tactical classes...")
    intelligence_model.train()
    
    for epoch in range(num_epochs):
        cumulative_epoch_loss = 0.0
        for i, (b_spectral, b_kinematic, b_labels) in enumerate(intel_loader):
            optimizer.zero_grad()
            
            # Multimodal forward pass
            inference_logits, _ = intelligence_model(b_spectral, b_kinematic)
            
            optimization_loss = objective_function(inference_logits, b_labels)
            optimization_loss.backward()
            optimizer.step()
            
            cumulative_epoch_loss += optimization_loss.item()
        
        avg_loss = cumulative_epoch_loss / len(intel_loader)
        print(f"  [Epoch {epoch+1}/{num_epochs}] Curriculum Loss: {avg_loss:.4f}")

    # 4. Intelligence Persistence
    os.makedirs("results", exist_ok=True)
    persistence_path = "results/tactical_intelligence_weights.pt"
    torch.save(intelligence_model.state_dict(), persistence_path)
    print(f"[TRAIN-IO] Tactical weights persisted to: {persistence_path}")
    
    return intelligence_model


if __name__ == "__main__":
    execute_intelligence_training_entrypoint()
