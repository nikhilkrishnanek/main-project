"""
Realistic Radar Noise and Clutter Modeling
==========================================

This module provides high-fidelity stochastic models for:
1. AWGN (Additive White Gaussian Noise): Receiver thermal noise.
2. Clutter: Non-Gaussian statistical models (Weibull, K-distribution) for environmental echoes.
3. Interference: Narrowband and broadband jamming signals.

Author: Radar Signal Processing Expert
"""

import numpy as np
from typing import Optional

def add_awgn(signal: np.ndarray, snr_db: float) -> np.ndarray:
    """
    Adds Additive White Gaussian Noise (AWGN) to the signal.
    """
    sig_power = np.mean(np.abs(signal)**2)
    noise_power = sig_power / (10**(snr_db / 10))
    
    # Complex noise for IQ signals
    noise = np.sqrt(noise_power/2) * (np.random.randn(*signal.shape) + 1j * np.random.randn(*signal.shape))
    return signal + noise

def generate_clutter(n_samples: int, distribution: str = 'weibull', **kwargs) -> np.ndarray:
    """
    Generates non-Gaussian radar clutter.
    
    Distributions:
    - 'weibull': Common for sea/land clutter at low grazing angles.
    - 'k': Common for high-resolution sea clutter.
    - 'gaussian': Simple Rayleigh clutter.
    """
    if distribution == 'weibull':
        scale = kwargs.get('scale', 1.0)
        shape = kwargs.get('shape', 1.5)
        # Weibull magnitude, random phase
        mag = scale * np.random.weibull(shape, n_samples)
    elif distribution == 'k':
        # K-distribution is a Product Model (Gamma * Rayleigh)
        shape = kwargs.get('shape', 2.0)
        scale = kwargs.get('scale', 1.0)
        texture = np.random.gamma(shape, scale, n_samples)
        speckle = np.sqrt(0.5) * (np.random.randn(n_samples) + 1j * np.random.randn(n_samples))
        return np.sqrt(texture) * speckle
    else:
        # Rayleigh / Gaussian Clutter
        return np.sqrt(0.5) * (np.random.randn(n_samples) + 1j * np.random.randn(n_samples))
    
    phase = np.random.uniform(0, 2*np.pi, n_samples)
    return mag * np.exp(1j * phase)

def add_interference(signal: np.ndarray, interference_type: str = 'narrowband', **kwargs) -> np.ndarray:
    """
    Simulates electronic interference or jamming.
    """
    fs = kwargs.get('fs', 1.0)
    n = len(signal)
    t = np.arange(n) / fs
    
    if interference_type == 'narrowband':
        freq = kwargs.get('freq', 0.1 * fs)
        amp = kwargs.get('amp', 1.0)
        jammer = amp * np.exp(1j * 2 * np.pi * freq * t)
    elif interference_type == 'sweep':
        # Swept jammer
        f0 = kwargs.get('f0', 0)
        f1 = kwargs.get('f1', fs/2)
        jammer = np.exp(1j * 2 * np.pi * (f0 * t + (f1-f0)/(2*t[-1]) * t**2))
    else:
        jammer = np.zeros_like(signal)
        
    return signal + jammer
