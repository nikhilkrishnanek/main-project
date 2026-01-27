"""
Optical Frequency Comb (OFC) Generation
=======================================

This module simulates the generation of an Optical Frequency Comb, 
providing a coherent multi-wavelength source for multi-band radar.

Mathematical Model:
E(t) = Sum( A_n * exp(j * (2*pi*(f0 + n*df)*t + phi_n)) )

Features:
- Flat-top comb simulation.
- Kerr-comb (Soliton) spectral envelope modeling.
- Phase noise correlation across lines.

Author: Photonic Radar Researcher
"""

import numpy as np
from typing import Tuple, List

def generate_flat_comb(
    n_lines: int, 
    spacing_hz: float, 
    center_freq_hz: float, 
    fs: float, 
    duration: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates a phase-coherent flat-top optical frequency comb.
    """
    n_samples = int(fs * duration)
    t = np.arange(n_samples) / fs
    
    # Complex envelope
    e_total = np.zeros(n_samples, dtype=complex)
    
    for n in range(-n_lines//2, n_lines//2):
        freq = center_freq_hz + n * spacing_hz
        # In a real comb, phases are locked. We simulate small residuals.
        phase_offset = np.random.normal(0, 0.01) 
        e_total += np.exp(1j * (2 * np.pi * freq * t + phase_offset))
    
    return t, e_total / np.sqrt(n_lines)

def generate_soliton_comb(
    n_lines: int,
    spacing_hz: float,
    center_freq_hz: float,
    fs: float,
    duration: float,
    comb_width_hz: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates a Kerr-comb with a sech^2 spectral envelope (Dissipative Kerr Soliton).
    """
    n_samples = int(fs * duration)
    t = np.arange(n_samples) / fs
    e_total = np.zeros(n_samples, dtype=complex)
    
    for n in range(-n_lines//2, n_lines//2):
        freq = center_freq_hz + n * spacing_hz
        # spectral weight (sech^2)
        weight = 1.0 / np.cosh((n * spacing_hz) / (comb_width_hz / 2))
        e_total += weight * np.exp(1j * (2 * np.pi * freq * t))
        
    return t, e_total / np.linalg.norm([1.0/np.cosh(i*spacing_hz/(comb_width_hz/2)) for i in range(-n_lines//2, n_lines//2)])

if __name__ == "__main__":
    t, e = generate_flat_comb(10, 25e9, 193.1e12, 1e12, 1e-9)
    print(f"Generated OFC with {len(t)} samples. Peak power: {np.max(np.abs(e)**2):.2f}")
