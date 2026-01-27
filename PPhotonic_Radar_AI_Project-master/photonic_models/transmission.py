"""
WDM / MDM Transmission Channel
==============================

Simulates multi-wavelength and multi-mode fiber transport.
Models:
1. Modal Dispersion (Differential Mode Delay).
2. Inter-modal and Inter-wavelength Cross-talk.
3. Nonlinear Fiber Effects (Simplified Kerr effect).

Author: Photonic Radar Researcher
"""

import numpy as np
from typing import List, Dict

def simulate_wdm_channel(
    signals: List[np.ndarray], 
    line_spacing_hz: float, 
    fiber_length_km: float, 
    cross_talk_db: float = -25.0
) -> List[np.ndarray]:
    """
    Simulates a Wavelength Division Multiplexed (WDM) channel with cross-talk.
    """
    n_channels = len(signals)
    xt_linear = 10**(cross_talk_db / 10)
    
    outputs = []
    for i in range(n_channels):
        # Start with the main channel signal
        total_sig = signals[i] * (1.0 - (n_channels-1) * xt_linear)
        
        # Add leakage from neighbors
        for j in range(n_channels):
            if i != j:
                total_sig += signals[j] * xt_linear
        
        outputs.append(total_sig)
        
    return outputs

def apply_modal_dispersion(
    signal: np.ndarray, 
    mode_index: int, 
    dmd_ps_km: float, 
    length_km: float, 
    fs: float
) -> np.ndarray:
    """
    Applies Differential Mode Delay (DMD) for MDM systems.
    """
    tau = mode_index * dmd_ps_km * length_km * 1e-12
    # Reuse TTD shift logic or simple roll if fs is high enough
    n_shift = int(tau * fs)
    if n_shift == 0: return signal
    
    return np.roll(signal, n_shift)

def fiber_nonlinear_phase_shift(
    signal: np.ndarray, 
    gamma: float = 1.3, 
    length_km: float = 10.0
) -> np.ndarray:
    """
    Simple Self-Phase Modulation (SPM) modeling.
    Phase shift proportional to optical power.
    """
    power = np.abs(signal)**2
    phi_nl = gamma * power * length_km
    return signal * np.exp(1j * phi_nl)
