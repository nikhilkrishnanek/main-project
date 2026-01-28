"""
Tactical Intelligence Training Orchestrator
===========================================

This module automates the supervised learning curriculum for the 
TacticalHybridClassifier. It manages the full training lifecycle, including 
stochastic gradient descent, multimodal dataset loading, and standardized 
performance benchmarking.

Performance Benchmarks:
-----------------------
- Precision, Recall, and F1-score for each tactical class.
- Cross-Entropy Loss optimization.
- Top-1 Classification Accuracy.

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
import numpy as np


def execute_intelligence_training_curriculum(num_epochs: int = 15, 
                                            training_batch_size: int = 32,
                                            learning_rate: float = 0.001):
    """
    Executes a complete training cycle for the radar intelligence model.
    """
    # 1. Synthesize Multimodal Training Data
    # Configuring the stochastic physics engine for data generation
    generation_config = {"chirp_duration_s": 0.05, "sampling_rate_hz": 1e5}
    data_engine = RadarDatasetGenerator(generation_config)
    
    print("[TRAIN-INTEL] Synthesizing high-fidelity tactical dataset...")
    raw_data = data_engine.generate_batch(samples_per_class=120)
    
    training_set = TensorDataset(
        raw_data["spectrograms"], 
        raw_data["time_series"], 
        raw_data["labels"]
    )
    data_loader = DataLoader(training_set, batch_size=training_batch_size, shuffle=True)
    
    # 2. Architect Model & Optimization Function
    target_labels = get_tactical_classes()
    intelligence_model = initialize_tactical_model(num_target_classes=len(target_labels))
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(intelligence_model.parameters(), lr=learning_rate)
    
    # 3. Supervised Optimization Loop
    print(f"[TRAIN-INTEL] Initiating curriculum for {len(target_labels)} tactical classes.")
    intelligence_model.train()
    
    for epoch in range(num_epochs):
        cumulative_loss = 0.0
        correct_identifications = 0
        total_samples = 0
        
        for spectral_maps, kinematic_series, ground_truth_labels in data_loader:
            optimizer.zero_grad()
            
            # Unpack multimodal outputs (logits + attention)
            prediction_logits, _ = intelligence_model(spectral_maps, kinematic_series)
            
            optimization_loss = criterion(prediction_logits, ground_truth_labels)
            optimization_loss.backward()
            optimizer.step()
            
            cumulative_loss += optimization_loss.item()
            _, predicted_indices = prediction_logits.max(1)
            total_samples += ground_truth_labels.size(0)
            correct_identifications += predicted_indices.eq(ground_truth_labels).sum().item()
            
        epoch_accuracy = 100. * correct_identifications / total_samples
        avg_loss = cumulative_loss / len(data_loader)
        
        print(f"  [Epoch {epoch+1:02d}/{num_epochs:02d}] Loss: {avg_loss:.4f} | Accuracy: {epoch_accuracy:.2f}%")
        
    return intelligence_model


def benchmark_intelligence_performance(intelligence_model: nn.Module, 
                                       evaluation_samples: int = 40):
    """
    Performs standardized tactical benchmarking of the trained model.
    """
    print("\n[EVAL-INTEL] Commencing rigorous performance benchmarking...")
    intelligence_model.eval()
    
    generation_config = {"chirp_duration_s": 0.05, "sampling_rate_hz": 1e5}
    data_engine = RadarDatasetGenerator(generation_config)
    validation_data = data_engine.generate_batch(samples_per_class=evaluation_samples)
    
    with torch.no_grad():
        # Multimodal validation pass
        logits, _ = intelligence_model(validation_data["spectrograms"], validation_data["time_series"])
        _, predicted_indices = logits.max(1)
        ground_truth = validation_data["labels"]
        
    correct_count = predicted_indices.eq(ground_truth).sum().item()
    total_count = ground_truth.size(0)
    final_accuracy = 100. * correct_count / total_count
    
    print(f"[EVAL-INTEL] Result Summary | Total: {total_count} | Correct: {correct_count} | Accuracy: {final_accuracy:.2f}%")
    
    # Detailed classification report
    from sklearn.metrics import classification_report
    target_names = get_tactical_classes()
    print("\n[EVAL-INTEL] TACTICAL CLASSIFICATION REPORT:")
    print(classification_report(ground_truth.numpy(), predicted_indices.numpy(), target_names=target_names))


if __name__ == "__main__":
    # Integration smoke test
    model = execute_intelligence_training_curriculum(num_epochs=5)
    benchmark_intelligence_performance(model)
    
    # Secure model persistence
    os.makedirs("results", exist_ok=True)
    persistence_path = "results/tactical_intelligence_v1.pt"
    torch.save(model.state_dict(), persistence_path)
    print(f"\n[IO-INTEL] Securely persisted model to: {persistence_path}")
