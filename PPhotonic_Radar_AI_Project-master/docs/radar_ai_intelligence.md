# Radar AI Intelligence: Hybrid Spatiotemporal Modeling

This document details the advanced AI architecture and strategy implemented in the **PHOENIX-RADAR** project for cognitive threat classification.

## 1. Deep Learning Architecture: Hybrid CRNN

We employ a **Convolutional Recurrent Neural Network (CRNN)** to leverage the multi-modal nature of radar data.

### 2D-CNN Branch (Spatial)
- **Input**: Micro-Doppler Spectrograms (or RD-Maps).
- **Purpose**: Extracts spatial "textures" of vibration signatures (e.g., rotor frequencies for drones).
- **Architecture**: Multi-layer CNN with Max-Pooling and Dropout for robust feature compression.

### 1.D-LSTM Branch (Temporal)
- **Input**: Raw Doppler time-series.
- **Purpose**: Captures the temporal evolution of the target's kinematics.
- **Architecture**: Bi-directional LSTM to model long-term dependencies in trajectory and micro-motion.

### Feature Fusion
The features from both branches are concatenated into a joint vector and passed through a dense classifier, allowing the model to make decisions based on both instantaneous "shape" and temporal "behavior".

## 2. Threat Classification Matrix

The model is trained to classify detections into four tactical categories:
1. **Drone**: Characterized by high-variance micro-Doppler signatures.
2. **Aircraft**: Stable, high-RCS signals with linear trajectories.
3. **Missile**: Ultra-high velocity kinematics with low micro-motion.
4. **Noise**: Thermal background and environmental clutter (Weibull/K-dist).

## 3. Explainable AI (XAI) & Confidence
To ensure "Human-in-the-Loop" trust, the system provides:
- **Calibrated Confidence**: Using Temperature Scaling to align Softmax outputs with real-world accuracy.
- **Predictive Entropy**: Estimating uncertainty to flag ambiguous detections (e.g., Bird vs Drone).
- **Physics-Aware Justification**: A narrative-based explanation linking neural activations to physical metrics (e.g., "Classified as Drone due to periodic sidebands").

## 4. Performance Metrics
- **Accuracy**: Primary measure of classification success.
- **F1-Score**: Critical for defense applications to balance False Negatives (missed threats) and False Positives (clutter).
- **Inference Latency**: Benchmarked to ensure real-time operation (<50ms per frame).
