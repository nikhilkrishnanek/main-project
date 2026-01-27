"""
Radar Metrics Module
===================

Provides functions to calculate theoretical and estimated system performance metrics.

Metrics:
1. Range Resolution (Delta R)
2. Velocity Resolution (Delta v)
3. SNR Estimation (from Data)

Author: Principal Photonic Radar Scientist
"""

import numpy as np
from dataclasses import dataclass

@dataclass
class RadarPerformance:
    range_resolution_m: float
    velocity_resolution_m_s: float
    max_range_m: float
    max_velocity_m_s: float

def calculate_theoretical_metrics(
    bandwidth: float, 
    chirp_duration: float,
    carrier_freq: float,
    n_chirps: int,
    fs: float
) -> RadarPerformance:
    """
    Calculates theoretical radar performance limits based on configuration.
    
    Delta R = c / (2 * B)
    Delta v = lambda / (2 * T_integration)
    """
    c = 3e8
    wavelength = c / carrier_freq
    total_integration_time = n_chirps * chirp_duration
    
    # 1. Resolutions
    delta_r = c / (2 * bandwidth)
    delta_v = wavelength / (2 * total_integration_time)
    
    # 2. Maximums (Nyquist limits)
    # Max Range: R_max = fs * c / (2 * slope)
    slope = bandwidth / chirp_duration
    max_r = (fs * c) / (2 * slope)
    
    # Max Velocity: v_max = lambda / (4 * PRI)
    # PRI approx equal to chirp_duration (assuming 100% duty cycle)
    max_v = wavelength / (4 * chirp_duration)
    
    return RadarPerformance(
        range_resolution_m=delta_r,
        velocity_resolution_m_s=delta_v,
        max_range_m=max_r,
        max_velocity_m_s=max_v
    )

def estimate_snr(rd_map_log: np.ndarray, peak_threshold_db: float = 10.0) -> float:
    """
    Estimates SNR from a Range-Doppler Map (Log Scale).
    
    Method:
    1. Identify Peak Signal Power (max value).
    2. Estimate Noise Floor (median or mean of lower quartile).
    3. SNR = Peak - NoiseFloor.
    """
    peak_power = np.max(rd_map_log)
    
    # Robust Noise Floor Estimation
    # Assume target occupies a small fraction of the map.
    # We use the median of the map as a proxy for noise floor
    # because targets are sparse outliers.
    noise_floor = np.median(rd_map_log)
    
    snr = peak_power - noise_floor
    
    return float(snr)

if __name__ == "__main__":
    # Test
    perf = calculate_theoretical_metrics(
        bandwidth=4e9,
        chirp_duration=10e-6,
        carrier_freq=10e9,
        n_chirps=64,
        fs=20e9
    )
    print(f"Metrics: {perf}")
    
    # Test SNR
    dummy_map = np.random.normal(0, 1, (100, 100)) # Noise
    dummy_map[50, 50] = 50.0 # Signal
    snr = estimate_snr(dummy_map)
    print(f"Est SNR: {snr:.2f} dB")
