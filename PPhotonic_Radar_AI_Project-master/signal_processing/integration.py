"""
Radar Temporal Integration and Clutter Mitigation
==================================================

This module provides specialized algorithms for processing radar data across 
the temporal dimension (Slow-Time or Frame-to-Frame) to enhance detection 
performance and suppress undesired static returns.

Module Features:
----------------
1. Coherent Integration: Complex summation across multiple pulses to maximize 
   SNR gain (Phasor aggregation).
2. Non-Coherent Integration (NCI): Power-domain averaging across multiple processing 
   frames to suppress stochastic noise noise variance.
3. Moving Target Indicator (MTI): High-pass filtering (Frame-differencing) to 
   cancel stationary clutter (zero Doppler) while preserving moving targets.

Author: Senior Signal Processing Engineer (Radar Systems Analyst)
"""

import numpy as np
from typing import List, Optional

def perform_coherent_integration(pulse_data_matrix: np.ndarray) -> np.ndarray:
    """
    Executes Coherent Integration (Complex-Mean) across the slow-time dimension.
    
    Processing Gain (SNR Improvement) ~ 10 * log10(num_pulses) dB.
    Requires phase stability (coherence) across the integration interval.
    """
    # Mean reduction across the pulse (slow-time) axis (Axis 0)
    return np.mean(pulse_data_matrix, axis=0)

def perform_non_coherent_integration(power_frame_sequence: List[np.ndarray]) -> np.ndarray:
    """
    Executes Non-Coherent Integration (Power-Mean) across multiple Range-Doppler maps.
    
    SNR improvement scales roughly with sqrt(N_frames). NCI is used when phase 
    coherence cannot be maintained over long durations.
    """
    if not power_frame_sequence:
        return np.array([])
    
    # Statistical averaging in the linear power domain
    return np.mean(power_frame_sequence, axis=0)

def clutter_cancellation_filter(current_intensity_db: np.ndarray, 
                                previous_intensity_db: np.ndarray) -> np.ndarray:
    """
    Implementation of a Single-Delay Line Canceler (SDLC) for clutter suppression.
    
    Subtracts subsequent intensity maps in the power domain to nullify 
    returns from stationary obstacles (e.g., terrain, buildings).
    """
    # Transform from logarithmic (dB) to linear power scale
    power_current = 10**(current_intensity_db / 10.0)
    power_previous = 10**(previous_intensity_db / 10.0)
    
    # Linear differencing
    cancellation_residual = power_current - power_previous
    
    # Numerical stability: Clip negative power to floor level
    cancellation_residual[cancellation_residual < 0] = 1e-12 
    
    return 10 * np.log10(cancellation_residual + 1e-12)

class FrameAccumulator:
    """
    State-management utility for real-time Temporal Integration buffers.
    """
    def __init__(self, capacity: int = 5):
        """
        Args:
            capacity: Number of recent frames to retain for non-coherent averaging.
        """
        self.capacity = capacity
        self.frame_buffer = []
        
    def add_frame(self, new_power_frame: np.ndarray) -> np.ndarray:
        """Adds a new frame to the circular buffer and returns the integrated output."""
        self.frame_buffer.append(new_power_frame)
        if len(self.frame_buffer) > self.capacity:
            self.frame_buffer.pop(0) # Maintain rolling window
            
        return perform_non_coherent_integration(self.frame_buffer)
    
    def reset_buffer(self):
        """Clears all historical frame data."""
        self.frame_buffer = []
