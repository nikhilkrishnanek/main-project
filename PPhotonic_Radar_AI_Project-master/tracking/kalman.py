"""
Radar Kalman Filter (Constant Acceleration)
===========================================

Implements a Constant Acceleration (CA) model for robust target state estimation.
Mathematical Framework:
1. State Vector (x): [Range, Velocity, Acceleration]^T
2. Measurement Vector (z): [Range, Velocity]^T
3. Transition Matrix (A): Newton's laws for constant acceleration.

Author: Principal Radar Tracking Scientist
"""

import numpy as np
from typing import Tuple

class KalmanFilter:
    def __init__(self, dt: float, process_noise: float = 0.05, measurement_noise: float = 0.5):
        """
        Initializes the CA Kalman Filter.
        
        Args:
            dt: Time step (sampling period).
            process_noise: Acceleration jitter (q).
            measurement_noise: Sensor uncertainty (r).
        """
        self.dt = dt
        
        # 1. State Vector (x) - [range, velocity, acceleration]
        self.x = np.zeros(3)
        
        # 2. State Transition Matrix (A)
        # r' = r + v*dt + 0.5*a*dt^2
        # v' = v + a*dt
        # a' = a
        self.A = np.array([
            [1.0, dt, 0.5 * dt**2],
            [0.0, 1.0, dt],
            [0.0, 0.0, 1.0]
        ])
        
        # 3. Measurement Matrix (H)
        # We measure range and velocity (from RD-FFT)
        self.H = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0]
        ])
        
        # 4. Process Noise Covariance (Q)
        # Assuming discrete-time white noise on acceleration
        q = process_noise
        self.Q = np.array([
            [(dt**4)/4, (dt**3)/2, (dt**2)/2],
            [(dt**3)/2, dt**2,     dt],
            [(dt**2)/2, dt,        1.0]
        ]) * q
        
        # 5. Measurement Noise Covariance (R)
        self.R = np.eye(2) * measurement_noise
        
        # 6. Error Covariance (P)
        self.P = np.eye(3) * 50.0  # High initial uncertainty
        
        # Statistics for Data Association
        self.S = np.eye(2)  # Innovation Covariance
        self.y = np.zeros(2) # Innovation (Residual)

    def predict(self) -> np.ndarray:
        """
        Prediction Step (A-priori state):
        x_k|k-1 = A * x_k-1
        P_k|k-1 = A * P * A^T + Q
        """
        self.x = self.A @ self.x
        self.P = self.A @ self.P @ self.A.T + self.Q
        return self.x

    def update(self, z: np.ndarray) -> np.ndarray:
        """
        Update Step (A-posteriori state):
        y = z - H * x_pred (Innovation)
        S = H * P_pred * H^T + R (Innovation Covariance)
        K = P_pred * H^T * S^-1 (Kalman Gain)
        x = x_pred + K * y
        P = (I - K * H) * P_pred
        """
        self.y = z - self.H @ self.x
        self.S = self.H @ self.P @ self.H.T + self.R
        
        # Kalman Gain
        K = self.P @ self.H.T @ np.linalg.inv(self.S)
        
        # State Update
        self.x = self.x + K @ self.y
        
        # Covariance Update (Joseph Form for numerical stability)
        I = np.eye(3)
        self.P = (I - K @ self.H) @ self.P @ (I - K @ self.H).T + K @ self.R @ K.T
        
        return self.x

    def mahalanobis_distance(self, z: np.ndarray) -> float:
        """
        Calculates the Mahalanobis distance between a measurement and the predicted state.
        Used for robust data association (Target ID validation).
        
        d^2 = y^T * S^-1 * y
        """
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        try:
            inv_S = np.linalg.inv(S)
            d2 = y.T @ inv_S @ y
            return float(np.sqrt(max(0, d2)))
        except np.linalg.LinAlgError:
            return 999.0 # Diverged

    def get_state(self) -> np.ndarray:
        return self.x

    def get_velocity(self) -> float:
        return float(self.x[1])

    def get_acceleration(self) -> float:
        return float(self.x[2])
