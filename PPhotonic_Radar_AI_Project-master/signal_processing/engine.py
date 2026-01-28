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

    def process_frame(self, pulse_matrix: np.ndarray, accumulate: bool = True) -> np.ndarray:
        """
        Executes the full Range-Doppler transformation with multi-frame integration.
        
        pulse_matrix: shape (num_pulses, samples_per_pulse)
        accumulate: If True, uses NCI (Temporal Integration) to improve SNR.
        """
        from signal_processing.integration import FrameAccumulator
        
        if not hasattr(self, 'accumulator'):
            self.accumulator = FrameAccumulator(capacity=self.config.get('nci_frames', 5))
            
        num_pulses, samples = pulse_matrix.shape
        
        # 1. Fast-Time (Range) Processing
        # Apply optimal window (Dolph-Chebyshev/Taylor)
        win_fast = apply_window(np.ones(samples), method=self.window_type)
        windowed_pulses = pulse_matrix * win_fast[np.newaxis, :]
        
        # FFT and Power scaling (1/N)
        range_fft = np.fft.fft(windowed_pulses, n=self.n_fft_range, axis=1) / samples
        
        # 2. Slow-Time (Doppler) Processing
        # Apply slow-time windowing
        win_slow = apply_window(np.ones(num_pulses), method=self.window_type)
        windowed_doppler = range_fft * win_slow[:, np.newaxis]
        
        # FFT and Power scaling (1/M)
        rd_complex = np.fft.fft(windowed_doppler, n=self.n_fft_doppler, axis=0) / num_pulses
        
        # Shift zero frequency to center
        rd_shifted = np.fft.fftshift(rd_complex, axes=0)
        
        # 3. Power Extraction (Linear domain)
        rd_linear_power = np.abs(rd_shifted)**2
        
        # 4. Multi-Frame Integration (NCI)
        if accumulate:
            rd_final_power = self.accumulator.add_frame(rd_linear_power)
        else:
            rd_final_power = rd_linear_power
            
        # 5. Log Transformation (dB)
        rd_db = 10 * np.log10(rd_final_power + 1e-12)
        
        return rd_db, rd_final_power

def create_rd_map_explicit(signal: np.ndarray, num_pulses: int, samples_per_pulse: int, config: Dict) -> np.ndarray:
    """
    Convenience wrapper for the explicit RD-FFT pipeline.
    """
    engine = RadarDSPEngine(config)
    # Reshape 1D signal into Pulse Matrix
    pulse_matrix = signal.reshape(num_pulses, samples_per_pulse)
    return engine.process_frame(pulse_matrix)
