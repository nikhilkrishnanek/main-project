# Tracking Intelligence: Multi-Target State Estimation

This document justifies and explains the Multi-Target Tracking (MTT) logic implemented in the **PHOENIX-RADAR** project.

## 1. Why Use Kalman Filtering?

Radar measurements (Range and Doppler) are inherently noisy due to:
- Optical phase noise in the photonic transceiver.
- Stochastic environmental clutter.
- Sensor quantization errors.

The **Kalman Filter** provides the optimal Bayesian estimate of the target's true state by recursively balancing the prediction (physics-based) and the measurement (sensor-based).

### Prediction Step
We model the target's movement using a constant-velocity model:
$$x_{k|k-1} = \Phi x_{k-1}$$
$$P_{k|k-1} = \Phi P_{k-1} \Phi^T + Q$$

### Update Step
When a new detection arrives, we update the estimate:
$$K_k = P_{k|k-1} H^T (H P_{k|k-1} H^T + R)^{-1}$$
$$x_k = x_{k|k-1} + K_k (z_k - H x_{k|k-1})$$

This process suppresses the "jitter" in tactical displays and provides accurate velocity estimation甚至 when the instantaneous Doppler measurement is degraded.

## 2. Multi-Target Data Association

In complex scenarios (e.g., Drone Swarms), the radar observes multiple simultaneous detections. **Data Association** is the process of linking these detections to persistent "Tracks".

### Global Nearest Neighbor (GNN)
We use a GNN approach to minimize the total Euclidean distance between predicted track positions and new detections. This ensures:
- **Persistent IDs**: The same drone is identified across multiple frames.
- **Track Coasting**: If a target is temporarily masked (missed detection), the Kalman filter "coasts" the track based on its previous velocity, allowing for seamless re-acquisition.

## 3. Improving Radar Intelligence

Tracking transforms "Detections" into "Intelligence":
1. **Trajectory Analysis**: Distinguishing between the erratic flight of a bird and the high-speed, direct path of a missile.
2. **Behavioral Prediction**: Projecting where a target will be in 5-10 seconds for automated counter-measure triggering.
3. **ID Persistence**: Maintaining situational awareness across a large surveillance volume without duplicate alerts.
