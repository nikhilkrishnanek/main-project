# 📡 PHOENIX-RADAR: Cognitive Photonic Radar with AI

![Status](https://img.shields.io/badge/Status-Operational-00f2ff)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Framework](https://img.shields.io/badge/Streamlit-App-red)

**A next-generation simulation and control platform for Cognitive Photonic Radar systems.**

This project simulates a high-frequency Photonic Radar system (FMCW/Heterodyne) and uses AI to classify targets based on their micro-Doppler signatures. It features a professional "Command Center" dashboard for real-time monitoring and control.

## 🌟 Key Features

- **Physics-Based Simulation**:
  - Photonic-generation of RF signals (Laser Phase Noise, Heterodyne Mixing).
  - Realistic FMCW Chirp, Delay, and Doppler modeling.
  - Simulation of **Drones**, **Birds**, **Aircraft**, **Missiles**, and **Clutter**.
  
- **Advanced Signal Processing**:
  - **Range-Doppler Maps** (2D FFT implementation).
  - **Micro-Doppler Spectrograms** (STFT).
  - Accurate CA-CFAR Detection.

- **AI/ML Integration**:
  - Multi-branch CNN Architecture (`RangeDoppler` + `Spectrogram` Fusion).
  - **Explainable AI (XAI)**: Strategic analysis and feature attribution for every detection.
  - Real-time Inference Engine with confidence metrics.

- **Research Benchmarking**:
  - Automated sensitivity curves (Pd vs SNR).
  - Operational reliability analysis (Pfa vs Threshold).
  - AI Robustness and Computational Latency benchmarks.

## 🏗️ Architecture

The project follows a modular 6-layer architecture designed for research scalability and defense readiness:

| Layer | Directory | Description |
|-------|-----------|-------------|
| **Core** | `core/` | System engine, telemetry, configuration, and security. |
| **Photonic** | `photonic/` | Physics-based modeling of signals, noise, and environmental factors. |
| **Signal** | `signal/` | Digital signal transformations, detection (CFAR), and feature extraction. |
| **AI** | `ai/` | Cognitive analysis, dual-stream neural networks, and XAI. |
| **Evaluation** | `evaluation/` | Research benchmarking, metric calculation, and performance tracking. |
| **UI** | `ui/` | Dashboard components, layout logic, and tactical visualization. |

For a detailed technical breakdown, see [ARCHITECTURE.md](docs/ARCHITECTURE.md).

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Linux/Windows/MacOS

### Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/your-repo/PPhotonic_Radar_AI_Project.git
    cd PPhotonic_Radar_AI_Project
    ```

2.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

### Usage

1.  **Launch the Dashboard**:
    ```bash
    streamlit run app.py
    ```
    This will open the "Command Center" in your default web browser (usually http://localhost:8501).

2.  **Simulation Mode**:
    - Select **Simulation** in the Sidebar.
    - Use the **Target Manager** to add/remove targets (e.g., set up a "Drone" at 50m with 10m/s velocity).
    - Click **RUN SIMULATION**.
    - Observe the real-time **Range-Doppler Map** and **AI Classification**.

### Verification

To verify the system integrity (useful for CI/CD or initial setup):
```bash
python3 tests/verify_pipeline.py
```

## 🛠️ Future Scope

- **Hardware-in-the-Loop (HIL)**: Integrate with RTL-SDR or SDRplay for real-time RF capture.
- **XAI (Explainable AI)**: Visualize Grad-CAM activation maps on the Range-Doppler input.
- **Tracking**: Implement Kalman Filter for multi-object tracking (MOT).

## 📄 License

This project is licensed under the MIT License.

---
*Designed for DRDO/Research Applications.*
