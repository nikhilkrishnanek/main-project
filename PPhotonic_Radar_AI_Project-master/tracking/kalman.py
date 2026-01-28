"""
Kinematic State Estimation: Constant Acceleration Kalman Filter
==============================================================

This module implements a linear Kalman Filter for recursive target state 
estimation. It utilizes a Constant Acceleration (CA) kinematic model to 
provide robust tracking of maneuverable targets.

Mathematical Notation:
----------------------
- State Vector (x): [Range, Velocity, Acceleration]^T
- Observation Vector (z): [Measured_Range, Measured_Velocity]^T
- State Transition (F): Maps state k-1 to state k.
- Process Noise (Q): Models uncertainty in the kinematic model (acceleration jitter).
- Measurement Noise (R): Models sensor uncertainty from the DSP pipeline.

Author: Senior Tracking & Fusion Engineer
"""

import numpy as np
from typing import Tuple


class KinematicKalmanFilter:
    """
    Recursive Bayesian estimator for target kinematics.
    """
    def __init__(self, 
                 sampling_period_s: float, 
                 process_noise_std: float = 0.05, 
                 measurement_noise_std: float = 0.5):
        """
        Initializes the Discrete-Time Kalman Filter.
        """
        self.dt = sampling_period_s
        
        # 1. State Vector (x) - [Radial Range (m), Radial Velocity (m/s), Radial Accel (m/s^2)]
        self.state_vector = np.zeros(3)
        
        # 2. State Transition Matrix (F)
        # s_k = s_{k-1} + v_{k-1}*dt + 0.5*a_{k-1}*dt^2
        # v_k = v_{k-1} + a_{k-1}*dt
        # a_k = a_{k-1}
        self.state_transition_matrix = np.array([
            [1.0, self.dt, 0.5 * self.dt**2],
            [0.0, 1.0,     self.dt],
            [0.0, 0.0,     1.0]
        ])
        
        # 3. Observation Matrix (H)
        # Maps state space to measurement space [Range, Doppler]
        self.observation_matrix = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0]
        ])
        
        # 4. Process Noise Covariance (Q)
        # Models the uncertainty in the constant acceleration assumption
        q_variance = process_noise_std**2
        dt2 = self.dt**2
        dt3 = self.dt**3
        dt4 = self.dt**4
        
        self.process_noise_covariance = np.array([
            [dt4/4.0, dt3/2.0, dt2/2.0],
            [dt3/2.0, dt2,     self.dt],
            [dt2/2.0, self.dt, 1.0]
        ]) * q_variance
        
        # 5. Measurement Noise Covariance (R)
        # Sensor accuracy as determined by the DSP/Ambiguity function
        self.measurement_noise_covariance = np.eye(2) * (measurement_noise_std**2)
        
        # 6. Error Covariance Matrix (P)
        # Represents the filter's certainty in the current state estimate
        self.error_covariance_matrix = np.eye(3) * 50.0  # High initial uncertainty
        
        # Residual Components (Innovation)
        self.innovation_vector = np.zeros(2)
        self.innovation_covariance = np.eye(2)

    def predict(self) -> np.ndarray:
        """
        Performs the Time Update (Prediction) step.
        """
        # x_pred = F * x
        self.state_vector = self.state_transition_matrix @ self.state_vector
        # P_pred = F * P * F^T + Q
        self.error_covariance_matrix = (self.state_transition_matrix @ 
                                       self.error_covariance_matrix @ 
                                       self.state_transition_matrix.T + 
                                       self.process_noise_covariance)
        return self.state_vector

    def update(self, measurement_vector: np.ndarray) -> np.ndarray:
        """
        Performs the Measurement Update (Correction) step.
        """
        # 1. Compute Innovation (y)
        self.innovation_vector = measurement_vector - self.observation_matrix @ self.state_vector
        
        # 2. Compute Innovation Covariance (S)
        self.innovation_covariance = (self.observation_matrix @ 
                                     self.error_covariance_matrix @ 
                                     self.observation_matrix.T + 
                                     self.measurement_noise_covariance)
        
        # 3. Compute Optimal Kalman Gain (K)
        # K = P * H^T * inv(S)
        kalman_gain = (self.error_covariance_matrix @ 
                       self.observation_matrix.T @ 
                       np.linalg.inv(self.innovation_covariance))
        
        # 4. Update State Estimate (A-posteriori)
        self.state_vector = self.state_vector + kalman_gain @ self.innovation_vector
        
        # 5. Update Error Covariance (Joseph Form for numerical stability)
        identity = np.eye(3)
        kh_matrix = kalman_gain @ self.observation_matrix
        self.error_covariance_matrix = ((identity - kh_matrix) @ 
                                       self.error_covariance_matrix @ 
                                       (identity - kh_matrix).T + 
                                       kalman_gain @ self.measurement_noise_covariance @ kalman_gain.T)
        
        return self.state_vector

    def calculate_mahalanobis_distance(self, measurement_vector: np.ndarray) -> float:
        """
        Computes the Mahalanobis distance between a candidate observation and 
        the predicted state. Used for Global Nearest Neighbor (GNN) association.
        """
        innovation = measurement_vector - self.observation_matrix @ self.state_vector
        try:
            inv_innovation_cov = np.linalg.inv(self.innovation_covariance)
            distance_squared = innovation.T @ inv_innovation_cov @ innovation
            return float(np.sqrt(max(0, distance_squared)))
        except np.linalg.LinAlgError:
            return 1e6 # Extreme distance for singular covariance

    def get_state_vector(self) -> np.ndarray:
        return self.state_vector

    def get_estimated_velocity(self) -> float:
        return float(self.state_vector[1])

    def get_estimated_acceleration(self) -> float:
        return float(self.state_vector[2])
