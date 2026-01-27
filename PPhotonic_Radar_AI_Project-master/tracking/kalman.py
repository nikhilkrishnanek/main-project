"""
Radar Kalman Filter
===================

Implements the Kalman Filter for robust target state estimation.
Filters sensor noise from Range and Doppler measurements and predicts 
future target states.

State Vector (x): [Range, Velocity]^T
Measurement Vector (z): [Range, Velocity]^T

Author: Radar Tracking Expert
"""

import numpy as np

class KalmanFilter:
    def __init__(self, dt: float, process_noise: float = 0.1, measurement_noise: float = 1.0):
        """
        dt: Time step between updates.
        """
        self.dt = dt
        
        # State: [range, velocity]
        self.x = np.array([0.0, 0.0])
        
        # State Transition Matrix (A)
        # range_new = range + velocity * dt
        # velocity_new = velocity
        self.A = np.array([[1.0, dt],
                           [0.0, 1.0]])
        
        # Measurement Matrix (H)
        # We measure both range and velocity directly from RD-Map
        self.H = np.eye(2)
        
        # Process Noise Covariance (Q)
        self.Q = np.eye(2) * process_noise
        
        # Measurement Noise Covariance (R)
        self.R = np.eye(2) * measurement_noise
        
        # Error Covariance Matrix (P)
        self.P = np.eye(2) * 10.0 # Initial uncertainty

    def predict(self):
        """
        Prediction Step:
        x_pred = A * x
        P_pred = A * P * A^T + Q
        """
        self.x = self.A @ self.x
        self.P = self.A @ self.P @ self.A.T + self.Q
        return self.x

    def update(self, z: np.ndarray):
        """
        Update Step:
        y = z - H * x_pred
        S = H * P_pred * H^T + R
        K = P_pred * H^T * S^-1
        x = x_pred + K * y
        P = (I - K * H) * P_pred
        """
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        self.x = self.x + K @ y
        self.P = (np.eye(2) - K @ self.H) @ self.P
        return self.x

    def get_state(self):
        return self.x

    def get_covariance(self):
        return self.P
