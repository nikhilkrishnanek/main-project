"""
Photonic True Time Delay (TTD) Beamforming
==========================================

This module simulates squint-free beamforming using photonic TTD.
Unlike electronic phase shifters, TTD maintains the same beam angle 
across ultra-wide bandwidths.

Mathematical Model:
S_delayed(t) = S(t - tau)
Beam Angle: theta = arcsin(c * tau / d)

Author: Photonic Radar Researcher
"""

import numpy as np
from scipy.signal import resample_poly
from typing import List

def apply_ttd_shift(signal: np.ndarray, tau: float, fs: float) -> np.ndarray:
    """
    Applies a True Time Delay using fractional delay filtering.
    """
    # Fractional delay in samples
    delay_samples = tau * fs
    
    # For simulation, we can use FFT-based shift for efficiency
    n = len(signal)
    freqs = np.fft.fftfreq(n, d=1/fs)
    shift_vector = np.exp(-1j * 2 * np.pi * freqs * tau)
    
    signal_fft = np.fft.fft(signal)
    return np.fft.ifft(signal_fft * shift_vector)

def simulate_ttd_array(
    signal: np.ndarray, 
    theta_deg: float, 
    d_m: float, 
    n_elements: int, 
    fs: float
) -> np.ndarray:
    """
    Simulates a linear array output for a given steering angle theta.
    
    theta_deg: Steering angle from boresight.
    d_m: Element spacing in meters.
    """
    c = 3e8
    theta_rad = np.deg2rad(theta_deg)
    
    outputs = []
    for i in range(n_elements):
        # tau_i = (i * d * sin(theta)) / c
        tau_i = (i * d_m * np.sin(theta_rad)) / c
        outputs.append(apply_ttd_shift(signal, tau_i, fs))
        
    return np.array(outputs)

def calculate_squint_error(theta_boresight: float, f_center: float, f_edge: float) -> float:
    """
    Calculates the beam squint error (in degrees) for electronic phase shifters.
    Photonic TTD has 0 squint error theoretically.
    """
    # sin(theta_f) = (f_center / f_edge) * sin(theta_boresight)
    sin_theta_f = (f_center / f_edge) * np.sin(np.deg2rad(theta_boresight))
    theta_f = np.rad2deg(np.arcsin(sin_theta_f))
    
    return float(np.abs(theta_f - theta_boresight))
