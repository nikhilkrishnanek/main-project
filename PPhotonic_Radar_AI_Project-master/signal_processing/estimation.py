"""
Radar Target Parameter Estimation
=================================

This module provides high-precision estimation of target state vectors (Range, 
Velocity) derived from detections in the Range-Doppler domain.

Algorithmic Approaches:
-----------------------
1. Sub-bin Peak Interpolation: Uses power-weighted centroiding to estimate target 
   location with precision exceeding the raw FFT bin resolution.
2. Canonical Radar Mapping: Translates spectral indices into physical SI units 
   (meters and m/s) based on the radar waveform physics.

Author: Senior Radar Systems Engineer
"""

import numpy as np
from typing import List, Tuple, Dict


def estimate_subpixel_centroid(range_doppler_intensity_db: np.ndarray, 
                               peak_index: Tuple[int, int], 
                               window_size: int = 3) -> Tuple[float, float]:
    """
    Estimates the sub-bin centroid of a spectral peak using power-weighted interpolation.
    
    This technique mitigates quantization errors inherent in the discrete FFT process.
    """
    row_idx, col_idx = peak_index
    num_rows, num_cols = range_doppler_intensity_db.shape
    
    half_win = window_size // 2
    row_start, row_end = max(0, row_idx - half_win), min(num_rows, row_idx + half_win + 1)
    col_start, col_end = max(0, col_idx - half_win), min(num_cols, col_idx + half_win + 1)
    
    sub_intensity_map = range_doppler_intensity_db[row_start:row_end, col_start:col_end]
    
    # Linearize power for accurate centroid weighting (dB -> Linear)
    linear_weights = 10**(sub_intensity_map / 10.0)
    
    # Create coordinate grids
    row_coords, col_coords = np.mgrid[row_start:row_end, col_start:col_end]
    
    # Calculate weighted average
    total_weight = np.sum(linear_weights)
    if total_weight == 0:
        return float(row_idx), float(col_idx) # Fallback
        
    row_centroid = np.sum(row_coords * linear_weights) / total_weight
    col_centroid = np.sum(col_coords * linear_weights) / total_weight
    
    return row_centroid, col_centroid


def map_spectral_indices_to_physics(fractional_peak_idx: Tuple[float, float], 
                                    radar_config: Dict) -> Dict[str, float]:
    """
    Translates fractional spectral indices to physical Range (m) and Velocity (m/s).
    
    Physics Model:
    - Range (R): Derived from the beat frequency (f_b) detected in the fast-time dimension.
      R = (c * f_b * T_chirp) / (2 * B)
    - Velocity (v): Derived from the Doppler shift (f_d) across pulses.
      v = (lambda * f_d) / 2
    """
    speed_of_light = 3e8
    doppler_bin, range_bin = fractional_peak_idx # Assumes (Doppler_Axis, Range_Axis)
    
    # Extract Waveform Parameters from Configuration
    sampling_rate_hz = radar_config.get('sampling_rate_hz', 20e9)
    sweep_bandwidth_hz = radar_config.get('sweep_bandwidth_hz', 4e9)
    chirp_duration_s = radar_config.get('chirp_duration_s', 10e-6)
    carrier_frequency_hz = radar_config.get('carrier_frequency_hz', 10e9)
    n_fft_range = radar_config.get('n_fft_range', 512)
    n_fft_doppler = radar_config.get('n_fft_doppler', 128)
    
    # 1. Precise Range Estimation
    # Compute beat frequency resolution per FFT bin
    range_bin_res_hz = sampling_rate_hz / n_fft_range
    f_beat_hz = range_bin * range_bin_res_hz
    
    # Canonical FMCW range equation
    sweep_slope = sweep_bandwidth_hz / chirp_duration_s
    range_m = (speed_of_light * f_beat_hz) / (2 * sweep_slope)
    
    # 2. Precise Velocity Estimation
    # Doppler resolution depends on the Pulse Repetition Frequency (PRF)
    # For back-to-back chirps, PRF = 1 / T_chirp
    prf_hz = 1.0 / chirp_duration_s
    doppler_bin_res_hz = prf_hz / n_fft_doppler
    
    # Account for zero-centering (fftshift)
    doppler_bin_offset = doppler_bin - (n_fft_doppler / 2.0)
    f_doppler_hz = doppler_bin_offset * doppler_bin_res_hz
    
    wavelength_m = speed_of_light / carrier_frequency_hz
    velocity_m_s = (wavelength_m * f_doppler_hz) / 2.0
    
    return {
        "range_m": float(range_m),
        "velocity_m_s": float(velocity_m_s),
        "beat_frequency_hz": float(f_beat_hz),
        "doppler_shift_hz": float(f_doppler_hz)
    }
