# Simulation Strategy: Real-Time Temporal Evolution

This document outlines the design strategy for the real-time simulation engine in the **PHOENIX-RADAR** project.

## 1. Temporal Discretization
To simulate physical motion, we discretize time into fixed intervals (Ticks).
- **Default Tick ($\Delta t$)**: 100ms (10 Hz Update Rate).
- Synchronizes the high-frequency Photonic layer with the lower-frequency Tracking and UI layers.

## 2. Physics-Informed Motion
Targets are no longer static "detections" but dynamic "entities" governed by kinematic models:
- **Kinematic State**: Includes Position, Velocity, and Acceleration.
- **Doppler Evolution**: As velocity changes (acceleration/maneuver), the IF beat frequency shifts dynamically, requiring the AI to perform temporal (LSTM) analysis.

## 3. Real-Time Pipeline Orchestration
The `SimulationOrchestrator` implements a strict pipeline order to ensure data integrity:

1. **Environmental Update**: Physics engine propagates target positions.
2. **Signal Synthesis**: Photonic layer generates RF signals based on instantaneous range/doppler.
3. **DSP/Inference**: Signal processing and AI classification identify threats.
4. **State Estimation**: Kalman trackers update persistent IDs.
5. **UI Refresh**: Dashboard visualizes the current tactical frame.

## 4. Performance Constraints
For radar applications, latency is critical. The system monitors:
- **Phase Latency**: Time spent in each architectural layer.
- **Jitter**: Variance in frame timing, which can degrade tracking accuracy.
- **Processor Load**: Balancing high-fidelity physics with real-time AI inference.
