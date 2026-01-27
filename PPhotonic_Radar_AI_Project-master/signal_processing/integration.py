"""
Radar Temporal Integration and MTI
==================================

This module provides logic for multi-frame processing:
1. Incoherent Integration: SNR improvement across frames.
2. MTI (Moving Target Indicator): Suppression of static clutter.
3. Pulse-to-Pulse correlation.

Author: Radar Signal Processing Expert
"""

import numpy as np
from typing import List, Optional

def incoherent_integration(frames: List[np.ndarray]) -> np.ndarray:
    """
    Performs incoherent integration (averaging magnitude) across multiple frames.
    Improves SNR by approx sqrt(N) where N is the number of frames.
    
    Frames should be logarithmic (dB) maps for robust averaging, or linear power.
    This implementation assumes Log-magnitude (dB) maps.
    """
    if not frames:
        return np.array([])
    
    # Average in linear power domain for accuracy
    linear_frames = [10**(f / 10) for f in frames]
    avg_linear = np.mean(linear_frames, axis=0)
    
    # Convert back to Log (dB)
    return 10 * np.log10(avg_linear + 1e-12)

def mti_filter(current_frame: np.ndarray, previous_frame: np.ndarray) -> np.ndarray:
    """
    Simple Moving Target Indicator (MTI) filter.
    Subtracts the previous frame from the current frame in the linear power domain.
    Effectively cancels out high-power static targets (clutter) while preserving moving targets.
    """
    # Convert to linear power
    p_curr = 10**(current_frame / 10)
    p_prev = 10**(previous_frame / 10)
    
    # Difference (Clutter cancellation)
    # We take the absolute difference or CLIP negative values if we care about magnitude
    diff = p_curr - p_prev
    diff[diff < 0] = 1e-12 # Threshold floor
    
    return 10 * np.log10(diff + 1e-12)

class FrameAccumulator:
    """
    Utility class to manage multi-frame state for real-time integration.
    """
    def __init__(self, capacity: int = 5):
        self.capacity = capacity
        self.buffer = []
        
    def add_frame(self, frame: np.ndarray) -> np.ndarray:
        self.buffer.append(frame)
        if len(self.buffer) > self.capacity:
            self.buffer.pop(0)
        return incoherent_integration(self.buffer)
    
    def clear(self):
        self.buffer = []
