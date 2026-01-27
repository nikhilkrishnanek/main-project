"""
Photonic Noise & Distortion Models
=================================

This module implements specific noise sources found in photonic radar systems.
It provides modular functions to degrade signals based on physical parameters.

Sources:
1. Thermal Noise (Receiver electronics)
2. Shot Noise (Photodetection quantum limit)
3. RIN (Laser intensity fluctuations)
4. Fiber Dispersion (Chromatic dispersion)

Author: Principal Photonic Radar Scientist
"""

import numpy as np
from dataclasses import dataclass

@dataclass
class NoiseConfig:
    """Configuration for Noise Sources."""
    temp_k: float = 290.0           # Temperature (Kelvin)
    load_resistance: float = 50.0   # Receiver Load (Ohms)
    rin_db_hz: float = -155.0       # RIN level (dB/Hz)
    bandwidth: float = 1e9          # Receiver Bandwidth (Hz)
    responsivity: float = 0.8       # PD Responsivity (A/W)
    fiber_dispersion: float = 17.0  # ps/(nm*km) for SMF-28
    fiber_length_km: float = 0.0    # Fiber link length

def add_thermal_noise(signal: np.ndarray, config: NoiseConfig, fs: float) -> np.ndarray:
    """
    Adds Thermal (Johnson) Noise to the voltage signal.
    Power = 4 * k * T * B * R (Voltage^2)
    """
    k_b = 1.380649e-23 # Boltzmann constant
    # Voltage Noise Variance = 4 k T B R
    var_v = 4 * k_b * config.temp_k * config.bandwidth * config.load_resistance
    sigma_v = np.sqrt(var_v)
    
    noise = np.random.normal(0, sigma_v, len(signal))
    return signal + noise

def add_shot_noise(signal_current: np.ndarray, config: NoiseConfig, fs: float) -> np.ndarray:
    """
    Adds Shot Noise to the photocurrent.
    I_shot^2 = 2 * q * I_avg * B
    
    Note: Signal here is assumed to be Current (Amps).
    Also, Shot noise is signal-dependent (Poisson), but for high flux
    we approximate as Gaussian with variance proportional to instantaneous current.
    """
    q = 1.60217663e-19 # Elementary charge
    
    # Instantaneous Shot Noise Variance
    # We take abs(current) because shot noise exists for any flow
    # Adding a small epsilon to avoid zero variance
    i_abs = np.abs(signal_current) + 1e-12
    var_i = 2 * q * i_abs * config.bandwidth
    
    # Generate non-stationary noise
    noise = np.random.normal(0, 1.0, len(signal_current)) * np.sqrt(var_i)
    return signal_current + noise

def add_rin_noise(optical_power_watts: float, config: NoiseConfig, n_samples: int, fs: float) -> np.ndarray:
    """
    Generates Relative Intensity Noise (RIN) current fluctuations.
    RIN is usually specified as dB/Hz.
    Var_RIN = RIN_linear * P_opt^2 * Bandwidth
    """
    rin_linear = 10**(config.rin_db_hz / 10)
    var_rin = rin_linear * (optical_power_watts**2) * config.bandwidth
    sigma_rin = np.sqrt(var_rin)
    
    # Converted to Current via Responsivity -> I = R * P
    # So Current Noise Sigma = R * Sigma_P
    sigma_i_rin = config.responsivity * sigma_rin
    
    noise = np.random.normal(0, sigma_i_rin, n_samples)
    return noise

def apply_fiber_dispersion(signal: np.ndarray, config: NoiseConfig, fs: float) -> np.ndarray:
    """
    Applies simplistic chromatic dispersion filter.
    Models frequency-dependent phase shift.
    
    This is a linear filter operation in frequency domain.
    """
    if config.fiber_length_km <= 0:
        return signal
        
    n = len(signal)
    freqs = np.fft.fftfreq(n, d=1/fs)
    
    # Dispersion Parameter D (ps/nm/km) needs conversion to Beta2 (s^2/km)
    # Beta2 approx -D * lambda^2 / (2 * pi * c)
    # Simplified phase shift for RF-over-Fiber (f_RF):
    # phi(f) = 0.5 * Beta2 * L * (2*pi*f)^2  <-- Quadratic phase
    # For this simulation, we use a simplified quadratic phase model per RF freq
    
    # Placeholder: A small quadratic phase error proportional to freq^2
    # Scaling factor for effect visibility
    alpha = 1e-25 * config.fiber_length_km 
    
    phase_shift = alpha * (2 * np.pi * freqs)**2
    
    H = np.exp(1j * phase_shift)
    
    sig_fft = np.fft.fft(signal)
    sig_filtered = np.fft.ifft(sig_fft * H)
    
    return np.real(sig_filtered) if np.isrealobj(signal) else sig_filtered
