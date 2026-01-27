# Technical Memo: Advanced Radar Signal Processing Logic

This document details the mathematical and structural logic of the **PHOENIX-RADAR** signal processing engine upgrade.

## 1. Range-Doppler FFT Pipeline
The engine separates radar processing into two orthogonal dimensions: **Fast-Time** (Range) and **Slow-Time** (Doppler).

### 1.1 Fast-Time (Range) Processing
- **Windowing**: A Taylor window is applied to each chirp to suppress range sidelobes (ensuring small targets near large ones are visible).
- **FFT**: A 1D-FFT is performed across the samples of a single chirp.
- **Result**: Conversion of time-delay (beat frequency) into Range Bins.

### 1.2 Slow-Time (Doppler) Processing
- **Pulse Accumulation**: Range-processed chips are stacked into a matrix.
- **FFT**: A 1D-FFT is performed across the pulses for each specific range bin.
- **Result**: Conversion of phase-shift between pulses into Doppler (Velocity) Bins.

---

## 2. Global Detection Strategy (CA-CFAR)
To maintain a Constant Probability of False Alarm (Pfa) in dynamic environments, we use a 2D Cell-Averaging CFAR.

### 2.1 The Thresholding Equation
The detection threshold $T$ is dynamically calculated:
$$T = \alpha \times P_{noise}$$

Where:
- $P_{noise}$ is the mean power in the **Training Ring**.
- $\alpha$ (Scaling Factor) is calculated based on the desired $P_{fa}$ and number of training cells $N$:
$$\alpha = N \cdot (P_{fa}^{-1/N} - 1)$$

### 2.2 Guard & Training Blocks
- **CUT (Cell Under Test)**: The center pixel.
- **Guard Band**: Prevents the target's own energy (sidelobes) from corrupting the noise estimate.
- **Training Ring**: The statistical sample of the background environment.

---

## 3. Stochastic Modeling
The upgrade replaces simple AWGN with tactical clutter profiles:

| Profile | Distribution | Use Case |
|---------|--------------|-----------|
| **Urban** | K-Distribution | Dense multipath, high-texture environments. |
| **Sea** | Weibull | Modeling high-resolution sea spikes and non-Rayleigh 'clutter edges'. |
| **Desert** | Rayleigh/Gaussian | Low-texture, thermal noise dominated environments. |

## 4. Processing Gain through Integration
### 4.1 Coherent Integration
Complex vector summation of $N$ pulses before magnitude detection. Provides a theoretical SNR gain of $10 \log_{10}(N)$.

### 4.2 Incoherent Integration
Magnitude-averaging across $M$ frames after detection. Provides an SNR gain approximately proportional to $\sqrt{M}$, used to stabilize intermittent detections and suppress transient noise spikes.
