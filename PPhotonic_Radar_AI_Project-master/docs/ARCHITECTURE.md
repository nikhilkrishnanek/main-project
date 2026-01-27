# PHOENIX-RADAR: System Architecture and Design Rationale

## 1. System Overview
The **PHOENIX-RADAR** (Cognitive Microwave Photonics Research System) is an advanced multi-layered software platform designed for the simulation, analysis, and cognitive processing of photonic-generated radar signals. The architecture leverages the broad bandwidth and low-loss transport properties of microwave photonics, coupled with state-of-the-art Deep Learning for automated target recognition (ATR).

## 2. Modular Decomposition
The project is structured into six logically isolated layers to ensure scalability, maintainability, and academic rigor.

### 2.1. Core Radar Logic (`core/`)
The **Core** layer serves as the central nervous system of the platform.
- **`engine.py`**: Orchestrates the data flow between the physics simulation, signal processing, and AI inference. It manages the "Radar Frame" lifecycle.
- **`config.py`**: Centralized configuration management using YAML, allowing researchers to tune system parameters without code modification.
- **`telemetry.py`**: Handles system health, startup integrity checks, and logging, essential for mission-critical defense applications.
- **`auth.py`**: Provides secure access control and authentication utilities.
- **`db.py` / `user_manager.py`**: Infrastructure components for user and session management.

### 2.2. Photonic Modeling Layer (`photonic/`)
This layer implements high-fidelity physical models of the microwave photonic transceiver.
- **`signals.py`**: Generates FMCW and heterodyne signals, modeling laser phase noise and optical beat frequencies.
- **`physics.py`**: Implements fiber-optic dispersion, coherence loss, and laser-linewidth-driven phase noise using stochastic Wiener processes.
- **`noise.py`**: Models fundamental photonic noise sources, including Relative Intensity Noise (RIN), Shot noise, and Thermal noise.
- **`environment.py`**: Simulates the target-channel interaction, including propagation delay, RCS-based attenuation, and Doppler shifts.
- **`scenarios.py`**: Provides standardized tactical benchmarks (e.g., Drone Swarm, Stealth Intruder) for repeatable research evaluations.

### 2.3. Digital Signal Processing Layer (`signal/`)
The **Signal** layer transforms raw optical-heterodyne outputs into interpretable tensors.
- **`transforms.py`**: Contains the mathematical core for Range-Doppler 2D-FFT and Micro-Doppler STFT transforms.
- **`features.py`**: Performs statistical feature extraction (Kurtosis, Skewness, Phase Variance) to augment the AI model's decision-making.
- **`detection.py`**: Implements classical radar detection algorithms like CA-CFAR (Cell-Averaging Constant False Alarm Rate) to isolate targets from background noise.

### 2.4. Artificial Intelligence Layer (`ai/`)
The **AI** layer provides cognitive interpretation of radar signatures.
- **`model.py`**: Defines a dual-stream CNN architecture that fuses spatial (Range-Doppler) and temporal (Spectrogram) features.
- **`inference.py`**: Optimization wrapper for real-time model execution.
- **`xai.py`**: **Explainable AI** module that maps neural network activations to physical radar parameters, providing "strategic narrative" to the operator.
- **`train.py`**: Automated training pipeline for adapting the model to new photonic noise profiles.

### 2.5. Evaluation & Analytics (`evaluation/`)
This layer provides the quantitative research metrics required for peer-reviewed publication and defense validation.
- **`metrics.py`**: Calculates theoretical resolution limits (Range/Velocity) and statistical performance estimators.
- **`benchmarking.py`**: Generates sensitivity curves (Pd vs SNR) and false alarm analysis (Pfa vs Threshold).
- **`radar_eval.py`**: High-level evaluation engine for end-to-end system testing.

### 2.6. User Interface & Visualization (`ui/`)
The **UI** layer translates complex physics data into a tactical Command Center.
- **`layout.py`**: Defines the dashboard structure and visualization layout.
- **`components.py`**: Reusable tactical components, including the 3D target manager and live signal monitors.

## 3. Design Rationale
- **Separation of Concerns**: By isolating physics (`photonic/`) from software logic (`core/`), the system can be easily transitioned from simulation to real-world Hardware-in-the-Loop (HIL) with minimal refactoring.
- **Research Standards**: The implementation of PD/PFA curves and XAI narratives aligns this project with standard IEEE radar signal processing benchmarks.
- **Defense Readiness**: The inclusion of telemetry, secure authentication, and tactical scenario generation makes the platform suitable for early-stage DRDO prototyping.
