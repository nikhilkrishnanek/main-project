"""
Radar Channel Environment
========================

This module simulates the physical environment, including target reflection and channel effects.
It implements the breakdown of the transmitted signal into multiple echoes, each affected by:
1. Time Delay (Range)
2. Phase Rotation (Doppler)
3. Attenuation (Radar Equation)
4. Channel Noise (AWGN)

Author: Principal Photonic Radar Scientist
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class Target:
    """Represents a physical radar target."""
    range_m: float          # Distance in meters
    velocity_m_s: float     # Radial velocity in m/s (positive = approaching)
    rcs_dbsm: float        # Radar Cross Section in dBsm
    description: str = "Unknown"

@dataclass
class ChannelConfig:
    """Configuration for the Environment."""
    c: float = 3e8          # Speed of light (m/s)
    carrier_freq: float = 10e9 # Carrier Frequency for Doppler calc
    noise_level_db: float = -50.0 # Noise floor relative to signal

def simulate_target_response(
    tx_signal: np.ndarray, 
    fs: float, 
    targets: List[Target], 
    config: ChannelConfig
) -> np.ndarray:
    """
    Simulates the receiver channel response for a list of targets.

    Args:
        tx_signal: The transmitted time-domain signal.
        fs: Sampling frequency (Hz).
        targets: List of Target objects.
        config: ChannelConfig parameters.

    Returns:
        rx_signal: The combined received signal with noise.
    """
    rx_signal = np.zeros_like(tx_signal, dtype=complex)
    n_samples = len(tx_signal)
    t = np.arange(n_samples) / fs
    
    # Pre-calculate noise
    noise_power = 10**(config.noise_level_db / 10)
    noise = np.random.normal(0, np.sqrt(noise_power/2), n_samples) + \
            1j * np.random.normal(0, np.sqrt(noise_power/2), n_samples)
    
    # Superposition of Echoes
    for tgt in targets:
        # 1. Calculate Parameters
        # Delay tau = 2R/c
        tau = 2 * tgt.range_m / config.c
        delay_samples = int(round(tau * fs))
        
        # Doppler Freq fd = 2v/lambda = 2*v*fc/c
        # Note: Standard physics definition often uses approaching as positive frequency shift.
        # We assume velocity is closing speed.
        f_doppler = (2 * tgt.velocity_m_s * config.carrier_freq) / config.c
        
        # Attenuation (Simplified Radar Equation Proportionality)
        # Power ~ RCS / R^4 -> Amplitude ~ sqrt(RCS) / R^2
        # We need a scaling factor to make simulation voltages realistic or normalized.
        # Here we use a relative scale factor.
        rcs_linear = 10**(tgt.rcs_dbsm / 10)
        # Avoid division by zero
        r_safe = max(tgt.range_m, 1.0)
        # Scale factor tuned for normalized output (demo purposes)
        # In rigorous sim, we would track absolute Watts.
        amplitude_scale = 1e4 * np.sqrt(rcs_linear) / (r_safe**2) 
        
        # 2. Construct Echo
        echo = np.zeros_like(tx_signal, dtype=complex)
        
        # Apply Time Delay
        if delay_samples < n_samples:
            # Shift signal
            # Valid region: [delay_samples : ]
            # Source region: [0 : n_samples - delay]
            valid_len = n_samples - delay_samples
            if valid_len > 0:
                echo[delay_samples:] = tx_signal[:valid_len]
                
        # Apply Doppler Phase Rotation
        # Multiply by exp(j * 2*pi * fd * t)
        doppler_mod = np.exp(1j * 2 * np.pi * f_doppler * t)
        echo *= doppler_mod
        
        # Apply Attenuation
        echo *= amplitude_scale
        
        # Accumulate
        rx_signal += echo
        
    # Add Channel Noise
    rx_signal += noise
    
    return rx_signal

if __name__ == "__main__":
    # Test
    fs = 1e6
    t = np.arange(1000) / fs
    tx = np.exp(1j * 2 * np.pi * 10e3 * t) # Simple tone
    cfg = ChannelConfig()
    tgts = [Target(range_m=100, velocity_m_s=30, rcs_dbsm=0)]
    
    rx = simulate_target_response(tx, fs, tgts, cfg)
    print(f"Generated RX signal: {rx.shape}, Max Amp: {np.max(np.abs(rx)):.4e}")
