"""
Radar Parameter Estimation
==========================

This module extracts physical target parameters (Range, Doppler) from signal peaks.
It includes algorithms for:
1. Centroid-based peak estimation.
2. Coordinate mapping from FFT bins to SI units.
3. Resolution-aware parameter bounding.

Author: Radar Signal Processing Expert
"""

import numpy as np
from typing import List, Tuple, Dict

def estimate_peak_centroid(rd_map: np.ndarray, peak_idx: Tuple[int, int], window_size: int = 3) -> Tuple[float, float]:
    """
    Estimates the sub-bin centroid of a peak using power-weighted averaging.
    Improves precision beyond the FFT bin resolution.
    """
    r_idx, d_idx = peak_idx
    M, N = rd_map.shape
    
    half = window_size // 2
    r_start, r_end = max(0, r_idx - half), min(M, r_idx + half + 1)
    d_start, d_end = max(0, d_idx - half), min(N, d_idx + half + 1)
    
    sub_map = rd_map[r_start:r_end, d_start:d_end]
    # Linearize power for weighting
    weights = 10**(sub_map / 10)
    
    r_coords, d_coords = np.mgrid[r_start:r_end, d_start:d_end]
    
    r_centroid = np.sum(r_coords * weights) / np.sum(weights)
    d_centroid = np.sum(d_coords * weights) / np.sum(weights)
    
    return r_centroid, d_centroid

def map_to_tactical(peak_idx: Tuple[float, float], config: Dict) -> Dict[str, float]:
    """
    Maps fractional FFT bins to physical Range (meters) and Velocity (m/s).
    
    Mathematical Justification:
    - Range: R = (c * f_beat) / (2 * slope)
    - Velocity: v = (lambda * f_doppler) / 2
    """
    c = 3e8
    r_bin, d_bin = peak_idx
    
    # Extract config parameters
    fs = config.get('fs', 20e9)
    bandwidth = config.get('bandwidth', 4e9)
    duration = config.get('duration', 10e-6)
    fc = config.get('fc', 10e9)
    n_fft_range = config.get('n_fft_range', 128)
    n_fft_doppler = config.get('n_fft_doppler', 128)
    
    # 1. Range Mapping
    # Frequency per bin in Fast-Time FFT
    df_range = fs / n_fft_range
    f_beat = r_bin * df_range
    slope = bandwidth / duration
    range_m = (c * f_beat) / (2 * slope)
    
    # 2. Doppler Mapping
    # Center frequency of Doppler FFT is 0 (due to fftshift)
    # Bin offset from center
    d_offset = d_bin - (n_fft_doppler / 2)
    # PRF is 1/duration (assuming consecutive chirps)
    prf = 1.0 / duration
    df_doppler = prf / n_fft_doppler
    f_doppler = d_offset * df_doppler
    
    wavelength = c / fc
    velocity_m_s = (wavelength * f_doppler) / 2
    
    return {
        "range_m": float(range_m),
        "velocity_m_s": float(velocity_m_s),
        "f_beat_hz": float(f_beat),
        "f_doppler_hz": float(f_doppler)
    }
