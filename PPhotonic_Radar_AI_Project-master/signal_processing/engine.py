"""
Modular Radar Digital Signal Processing (DSP) Engine
====================================================

This module provides an object-oriented interface for the radar signal processing 
pipeline. It orchestrates the transformation from raw time-domain pulse matrices 
to multi-dimensional spectral maps, incorporating windowing, FFTs, and temporal 
integration (NCI).

Processing Stages:
------------------
1. Fast-Time Matched Filtering/FFT: Range extraction.
2. Slow-Time FFT: Doppler velocity extraction.
3. Power Estimation: Conversion from complex-amplitude to power-spectral density (PSD).
4. Non-Coherent Integration (NCI): Multi-frame averaging to suppress stochastic 
   noise and enhance target detection sensitivity.

Author: Senior DSP Systems Architect
"""

import numpy as np
from typing import Dict, Tuple, Optional
from signal_processing.transforms import get_specialized_window


class RadarDSPEngine:
    """
    Stateful Radar Processor capable of multi-frame integration and 
    configurable spectral estimation.
    """
    def __init__(self, config: Dict):
        self.config = config
        self.n_fft_range = config.get('n_fft_range', 512)
        self.n_fft_doppler = config.get('n_fft_doppler', 128)
        self.window_type = config.get('window_type', 'taylor')

    def process_frame(self, pulse_data_matrix: np.ndarray, 
                      enable_integration: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Executes the dual-FFT Range-Doppler processing chain.
        
        Args:
            pulse_data_matrix: (num_pulses, samples_per_pulse) complex baseband data.
            enable_integration: If True, applies multi-frame NCI.
            
        Returns:
            (intensity_db, power_linear): The processed maps in both scales.
        """
        from signal_processing.integration import FrameAccumulator
        
        if not hasattr(self, 'accumulator'):
            # Initialize NCI buffer if not present
            self.accumulator = FrameAccumulator(capacity=self.config.get('nci_frames', 5))
            
        num_pulses, samples_per_pulse = pulse_data_matrix.shape
        
        # --- Stage 1: Fast-Time (Range) Processing ---
        # Generate apodization window to minimize range sidelobes (spectral leakage)
        range_window = get_specialized_window(samples_per_pulse, method=self.window_type, at=80)
        windowed_pulses = pulse_data_matrix * range_window[np.newaxis, :]
        
        # Execute Range FFT across the fast-time dimension (Axis 1)
        # Scaling by 1/N preserves average signal energy levels
        range_spectral_matrix = np.fft.fft(windowed_pulses, n=self.n_fft_range, axis=1) / samples_per_pulse
        
        # --- Stage 2: Slow-Time (Doppler) Processing ---
        # Apply Doppler windowing to suppress velocity sidelobes
        doppler_window = get_specialized_window(num_pulses, method=self.window_type, at=60)
        windowed_doppler = range_spectral_matrix * doppler_window[:, np.newaxis]
        
        # Execute Doppler FFT across the pulse index dimension (Axis 0)
        # Centering DC (Zero Doppler) for intuitive visualization
        rd_complex = np.fft.fft(windowed_doppler, n=self.n_fft_doppler, axis=0) / num_pulses
        rd_centered = np.fft.fftshift(rd_complex, axes=0)
        
        # --- Stage 3: Power Characteristic Extraction ---
        # Square-law detection (Linear Power)
        range_doppler_power_linear = np.abs(rd_centered)**2
        
        # --- Stage 4: Non-Coherent Integration (NCI) ---
        # Averaging power across time frames to increase Signal-to-Noise Ratio (SNR)
        if enable_integration:
            integrated_power = self.accumulator.add_frame(range_doppler_power_linear)
        else:
            integrated_power = range_doppler_power_linear
            
        # --- Stage 5: Intensity Normalization (Decibel Scale) ---
        range_doppler_intensity_db = 10 * np.log10(integrated_power + 1e-12)
        
        return range_doppler_intensity_db, integrated_power


def compute_tactical_spectral_map(signal_sequence: np.ndarray, 
                                  num_pulses: int, 
                                  samples_per_pulse: int, 
                                  processing_config: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """
    Procedural wrapper for standard RD processing of serialized pulse streams.
    """
    dsp_engine = RadarDSPEngine(processing_config)
    # Reshape 1D serialized byte/float stream into structured Pulse Matrix
    pulse_data_matrix = signal_sequence.reshape(num_pulses, samples_per_pulse)
    return dsp_engine.process_frame(pulse_data_matrix)
