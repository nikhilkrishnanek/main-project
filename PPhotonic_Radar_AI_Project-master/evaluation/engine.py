import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

def evaluate_pytorch(model, loader, device="cpu"):
    model.eval()
    all_preds = []
    all_y = []
    
    with torch.no_grad():
        for b_rd, b_spec, b_meta, b_y in loader:
            b_rd, b_spec, b_meta = b_rd.to(device), b_spec.to(device), b_meta.to(device)
            outputs = model(b_rd, b_spec, b_meta)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_y.extend(b_y.numpy())

    print(classification_report(all_y, all_preds))

    cm = confusion_matrix(all_y, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.savefig("results/confusion_matrix.png")
    
    Pd = np.trace(cm) / np.sum(cm)
    FAR = (np.sum(cm) - np.trace(cm)) / np.sum(cm)

    print(f"Probability of Detection (Pd): {Pd:.4f}")
    print(f"False Alarm Rate (FAR): {FAR:.4f}")
