# AI Radar Intelligence: Cognitive Classification & XAI

## 1. Hybrid spatiotemporal Architecture (CRNN)

Modern radar electronic support measures (ESM) require the ability to classify targets in complex, non-Gaussian environments. We implement a **Convolutional Recurrent Neural Network (CRNN)** that fuses spatial and temporal features:

### CNN Branch (Spatial Extraction)
- **Input**: 2D Range-Doppler maps or Spectrograms.
- **Method**: Residual neural blocks (ResNet-lite) identify textures in the RD-space, such as the spread of a drone rotor vs. the sharp peak of a missile.
- **Advantage**: Handles translation invariance and extracts multi-scale spatial features.

### LSTM Branch with Attention (Temporal Extraction)
- **Input**: 1D Doppler time-series.
- **Method**: Bidirectional LSTM layers with a **Bahdanau Attention Mechanism**.
- **Reasoning**: Not all time segments are equally important. Attention allows the model to "focus" on specific bursts or modulation cycles (e.g., when a rotor blade is perpendicular to the beam).

---

## 2. High-Fidelity Micro-Doppler Modeling

To train a robust model, we use physically-informed synthetic data:
- **Drone (Micro-Doppler)**: Simulated using the sum of $N$ phase-modulated carriers, representing $N$ rotors with variable blade lengths and RPMs.
- **Bird (Biological)**: Erratic amplitude modulation (AM) and low-frequency Doppler shifts simulating flapping wings.
- **Missile/Aircraft**: High-velocity bulk Doppler with rigid-body RCS profiles.

---

## 3. Explainable AI (XAI) for Critical Systems

In defense applications, "Black Box" AI is unacceptable. We provide two layers of transparency:

### Spatial Transparency (Grad-CAM)
Using **Gradient-weighted Class Activation Mapping**, we project the AI's "focus" back onto the RD-Map. This allows operators to verify if the AI detected the target or was tricked by background clutter.

### Statistical Reliability (MC-Dropout)
We estimate **Epistemic Uncertainty** using Monte Carlo Dropout. By running $N$ stochastic inferences, we calculate the entropy of the prediction. High entropy indicates the signature is ambiguous (e.g., a bird that looks like a drone), allowing the system to flag "Low Confidence" for human review.

---

## 4. Photonic Integration 

The AI layer is directly integrated with the **MWP (Microwave Photonic)** layer. Our synthetic data includes:
- **WDM Crosstalk**: High-frequency inter-modulations.
- **TTD Phase Noise**: Modeled from optical jitter.
- **Flat-Comb Aliasing**: Simulation of multi-wavelength signal interference.
