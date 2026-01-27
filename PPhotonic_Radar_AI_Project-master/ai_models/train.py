import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from signal.features import extract_features as get_all_features
from photonic.signals import generate_photonic_signal as generate_radar_signal
from ai.model import DualStreamRadarNet as build_pytorch_model
import cv2
import os

def create_pytorch_dataset(samples_per_class=50):
    classes = ["drone", "aircraft", "bird", "helicopter", "missile", "clutter"]
    rd_list, spec_list, meta_list, y_list = [], [], [], []

    print("Generating simulated photonic radar dataset...")
    for label, cls in enumerate(classes):
        for _ in range(samples_per_class):
            sig = generate_radar_signal(cls)
            rd, spec, meta, _ = get_all_features(sig)
            
            # Resize to match model input
            rd = cv2.resize(rd, (128, 128))
            spec = cv2.resize(spec, (128, 128))
            
            # Normalize
            rd = rd / (np.max(rd) + 1e-8)
            spec = spec / (np.max(spec) + 1e-8)
            
            rd_list.append(rd)
            spec_list.append(spec)
            meta_list.append(meta)
            y_list.append(label)

    return (
        torch.tensor(np.array(rd_list), dtype=torch.float32),
        torch.tensor(np.array(spec_list), dtype=torch.float32),
        torch.tensor(np.array(meta_list), dtype=torch.float32),
        torch.tensor(np.array(y_list), dtype=torch.long)
    )

def train_pytorch_model(epochs=10):
    rd, spec, meta, y = create_pytorch_dataset()
    dataset = TensorDataset(rd, spec, meta, y)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    model = build_pytorch_model(num_classes=6)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    print("Starting training...")
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for i, (b_rd, b_spec, b_meta, b_y) in enumerate(loader):
            optimizer.zero_grad()
            outputs = model(b_rd, b_spec, b_meta)
            loss = criterion(outputs, b_y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(loader):.4f}")

    # Save the model
    os.makedirs("results", exist_ok=True)
    torch.save(model.state_dict(), "results/radar_model_pytorch.pt")
    print("Model saved to results/radar_model_pytorch.pt")
    return model

if __name__ == "__main__":
    train_pytorch_model()
