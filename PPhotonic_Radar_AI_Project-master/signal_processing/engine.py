"""
Modular Radar Signal Processing Engine
=====================================

This module provides an explicit, stage-by-stage radar processing pipeline.
Separates fast-time and slow-time processing for clarity and research flexibility.

Pipeline Stages:
1. Range FFT (Fast-Time)
2. Doppler FFT (Slow-Time)
3. 2D Spectral Energy Calculation
4. Normalization and Log-conversion

Author: Radar Signal Processing Engineer
"""

import numpy as np
from typing import Dict, Tuple, Optional
from signal_processing.transforms import apply_window

class RadarDSPEngine:
    def __init__(self, config: Dict):
        self.config = config
        self.n_fft_range = config.get('n_fft_range', 512)
        self.n_fft_doppler = config.get('n_fft_doppler', 128)
        self.window_type = config.get('window_type', 'taylor')

    def process_frame(self, pulse_matrix: np.ndarray) -> np.ndarray:
        """
        Executes the full Range-Doppler transformation.
        
        pulse_matrix: shape (num_pulses, samples_per_pulse)
        """
        num_pulses, samples = pulse_matrix.shape
        
        # 1. Range Processing (Fast-Time)
        # Apply window along the fast-time dimension
        windowed_pulses = np.apply_along_axis(
            lambda x: apply_window(x, self.window_type), 
            axis=1, 
            arr=pulse_matrix
        )
        
        # Fast FFT along rows
        range_fft = np.fft.fft(windowed_pulses, n=self.n_fft_range, axis=1)
        
        # 2. Doppler Processing (Slow-Time)
        # Apply window along the slow-time dimension
        windowed_range = np.apply_along_axis(
            lambda x: apply_window(x, self.window_type),
            axis=0,
            arr=range_fft
        )
        
        # FFT along columns
        rd_complex = np.fft.fft(windowed_range, n=self.n_fft_doppler, axis=0)
        
        # Shift zero frequency to center
        rd_shifted = np.fft.fftshift(rd_complex, axes=0)
        
        # 3. Power Scaling
        # We return the magnitude magnitude squared (Power)
        rd_power = np.abs(rd_shifted)**2
        
        # 4. Log Transformation (dB)
        # 10 * log10(Power)
        rd_db = 10 * np.log10(rd_power + 1e-12)
        
        return rd_db

def create_rd_map_explicit(signal: np.ndarray, num_pulses: int, samples_per_pulse: int, config: Dict) -> np.ndarray:
    """
    Convenience wrapper for the explicit RD-FFT pipeline.
    """
    engine = RadarDSPEngine(config)
    # Reshape 1D signal into Pulse Matrix
    pulse_matrix = signal.reshape(num_pulses, samples_per_pulse)
    return engine.process_frame(pulse_matrix)
