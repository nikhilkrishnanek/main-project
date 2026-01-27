# Radar Tracking Theory & Logic

## 1. Kalman Filter (Constant Acceleration Model)

In complex military scenarios (e.g., drone maneuvering), a Constant Velocity (CV) model is insufficient. We implement a **Constant Acceleration (CA)** model to track targets through turns and speed changes.

### State Vector ($x$)
The state represents the physical parameters of the target at any time $k$:
$$x_k = [r, v, a]^T$$
where $r$ is Range, $v$ is Velocity, and $a$ is Acceleration.

### Prediction Step (Motion Model)
We project the current state into the future using Newton's laws of motion:
- $r_{k} = r_{k-1} + v_{k-1}\Delta t + \frac{1}{2}a_{k-1}\Delta t^2$
- $v_{k} = v_{k-1} + a_{k-1}\Delta t$
- $a_{k} = a_{k-1} + \epsilon$ (where $\epsilon$ is process noise)

**Equations:**
1. $\hat{x}_{k|k-1} = A x_{k-1}$
2. $P_{k|k-1} = A P_{k-1} A^T + Q$

### Update Step (Measurement Fusion)
When a detection $(z = [r_z, v_z]^T)$ is received, we fuse it with our prediction:
1. **Innovation ($y$):** Difference between measured and predicted.
2. **Innovation Covariance ($S$):** Uncertainty in the measurement space.
3. **Kalman Gain ($K$):** Weight given to the measurement vs the model.

**Equations:**
1. $y = z - H \hat{x}_{k|k-1}$
2. $S = H P_{k|k-1} H^T + R$
3. $K = P_{k|k-1} H^T S^{-1}$
4. $x_k = \hat{x}_{k|k-1} + K y$

---

## 2. Target Data Association (Mahalanobis Distance)

Instead of simple Euclidean distance, we use the **Mahalanobis Distance** to account for the uncertainty (covariance) of the track.

$$d^2 = y^T S^{-1} y$$

This "statistical distance" validates if a detection $(z)$ realistically belongs to Track $(i)$ by checking how many standard deviations it is from the prediction. We use an **Association Gate** (e.g., 3.5$\sigma$) to prevent matching with noise or clutter.

---

## 3. Track Lifecycle Management

To filter out false alarms and maintain tracks through signal fades, we implement a state machine:

| State | Condition | Action |
| :--- | :--- | :--- |
| **PROVISIONAL** | Initial detection | Needs 3 consecutive hits to "confirm". |
| **CONFIRMED** | High confidence | Actively displayed on the dashboard. |
| **COASTING** | Missed detection | Predict position using motion model; wait for re-acquisition. |
| **DELETED** | Too many misses | Remove from memory (Target lost or out of range). |

### Re-acquisition Logic
If a target enters **COASTING** (e.g., due to EMI or shadow), the radar "coasts" its state for up to 10 frames. If a matching detection appears within the Predicted Gate during this time, the track is restored to **CONFIRMED** instantly, preserving the Target ID.
