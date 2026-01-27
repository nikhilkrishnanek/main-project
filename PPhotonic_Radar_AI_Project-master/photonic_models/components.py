"""
Photonic Components: MZM and PD
===============================

Models the electro-optic modulation and photodetection chain.
- Mach-Zehnder Modulator (MZM): Transfer function and Bias control.
- Photodiode (PD): Responsivity and Shot noise.

Author: Photonic Radar Researcher
"""

import numpy as np

def mzm_modulate(
    optical_carrier: np.ndarray, 
    rf_signal: np.ndarray, 
    v_pi: float = 5.0, 
    v_bias: float = 2.5
) -> np.ndarray:
    """
    Simulates Mach-Zehnder Modulation.
    E_out = E_in * cos( (pi/2) * (V_rf + V_bias) / V_pi )
    """
    phase = (np.pi / 2) * (rf_signal + v_bias) / v_pi
    return optical_carrier * np.cos(phase)

def photodiode_detect(
    optical_field: np.ndarray, 
    responsivity: float = 0.8, 
    dark_current: float = 1e-9,
    add_shot_noise: bool = True
) -> np.ndarray:
    """
    Square-law detection and conversion to photocurrent.
    I = R * |E|^2 + I_dark + ShotNoise
    """
    # Linear power
    optical_power = np.abs(optical_field)**2
    current = responsivity * optical_power + dark_current
    
    if add_shot_noise:
        # Shot noise is Poisson, approximated as Gaussian for large currents
        # Var(I_shot) = 2 * q * I * BW
        q = 1.6e-19
        # Assuming bandwidth is normalized to samples for this simulation
        noise_std = np.sqrt(2 * q * np.mean(current))
        current += noise_std * np.random.randn(*current.shape)
        
    return current
