# Radar Simulation Strategy: Real-Time Execution Framework

## 1. Overview

The `SimulationOrchestrator` provides a high-fidelity, synchronous execution loop that mirrors a real-world radar hardware cycle. Each "tick" of the loop simulates a complete frame transmission, reception, and processing cycle.

## 2. The Execution Frame Loop

A single frame consists of five distinct phases, meticulously orchestrated to maintain real-time fidelity:

### Phase A: Kinematic Physics
The `KinematicEngine` updates the ground-truth state of all targets. We use a **Constant Acceleration (CA)** model:
$s_{k+1} = s_k + v_k \Delta t + \frac{1}{2} a_k \Delta t^2$
Supports sinusoidal maneuvers (drone hover) and high-G evasive maneuvers.

### Phase B: Photonic Generation
Translates ground-truth positions into raw IQ data. 
- **Doppler Mapping**: Velocity is converted to phase shifts in the signal.
- **WDM Noise**: Optical crosstalk and jitter are injected to simulate photonic frontend effects.

### Phase C: Signal Processing (DSP)
The `RadarDSPEngine` performs 2D FFT transformations to generate Range-Doppler maps.
- **CA-CFAR**: Logical detection identifies target centroids amidst clutter.

### Phase D: Multi-Target Tracking (MTT)
The `TrackManager` maintains target identities across frames.
- Uses **Mahalanobis Distance** for statistically robust association.
- Transitions tracks through `PROVISIONAL` -> `CONFIRMED` -> `COASTING` states.

### Phase E: Performance Monitoring
The `PerformanceMonitor` records the latency of each phase.
- Ensures the total cycle remains under the 100ms budget.
- Provides real-time "Effective FPS" metrics.

---

## 3. Integration with Dashboard

The simulation is designed to be consumed as a **Python Generator**:

```python
orchestrator = SimulationOrchestrator(config, initial_targets)
for frame_data in orchestrator.run_loop():
    # Update UI Components
    st.session_state.last_rd_map = frame_data['rd_map']
    st.session_state.active_tracks = frame_data['tracks']
    time.sleep(0.01) # Yield to UI thread
```

This ensures that the simulation logic remains decoupled from the visualization layer, allowing for high-performance execution without UI blocking.
