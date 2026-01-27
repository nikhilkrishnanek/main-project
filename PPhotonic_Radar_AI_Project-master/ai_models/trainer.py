"""
Radar AI Training and Evaluation Pipeline
=========================================

Automates the training process for the Hybrid CRNN architecture.
Features:
- Automated dataset generation.
- Scalable PyTorch training loop.
- Multi-metric evaluation (Accuracy, F1, Loss).

Author: Radar AI Engineer
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from ai_models.dataset_generator import RadarDatasetGenerator
from ai_models.architectures import get_hybrid_model
import numpy as np

def train_radar_intelligence(epochs=10, batch_size=32):
    # 1. Prepare Data
    gen_cfg = {"duration": 0.05, "fs": 1e5}
    generator = RadarDatasetGenerator(gen_cfg)
    
    print("Generating training dataset...")
    train_data = generator.generate_batch(samples_per_class=100)
    
    dataset = TensorDataset(
        train_data["spectrograms"], 
        train_data["time_series"], 
        train_data["labels"]
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # 2. Initialize Model
    model = get_hybrid_model(num_classes=4)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 3. Training Loop
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        correct = 0
        total = 0
        
        for specs, ts, labels in loader:
            optimizer.zero_grad()
            outputs = model(specs, ts)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
        acc = 100. * correct / total
        print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss/len(loader):.4f} | Acc: {acc:.2f}%")
        
    return model

def evaluate_model(model, num_samples=20):
    model.eval()
    gen_cfg = {"duration": 0.05, "fs": 1e5}
    generator = RadarDatasetGenerator(gen_cfg)
    test_data = generator.generate_batch(samples_per_class=num_samples)
    
    with torch.no_grad():
        outputs = model(test_data["spectrograms"], test_data["time_series"])
        _, predicted = outputs.max(1)
        labels = test_data["labels"]
        
    correct = predicted.eq(labels).sum().item()
    total = labels.size(0)
    print(f"\nFinal Evaluation | Total: {total} | Correct: {correct} | Accuracy: {100.*correct/total:.2f}%")
    
    # Simplified Confusion matrix print
    from sklearn.metrics import classification_report
    target_names = ["Drone", "Aircraft", "Missile", "Noise"]
    print("\nClassification Report:")
    print(classification_report(labels.numpy(), predicted.numpy(), target_names=target_names))

if __name__ == "__main__":
    trained_model = train_radar_intelligence(epochs=5)
    evaluate_model(trained_model)
    
    # Save model
    os.makedirs("results", exist_ok=True)
    torch.save(trained_model.state_dict(), "results/hybrid_radar_net.pt")
    print("Model saved to results/hybrid_radar_net.pt")
