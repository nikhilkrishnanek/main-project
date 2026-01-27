"""
Radar Signal Processing Transforms
=================================

This module provides the core DSP algorithms to transform raw IF signals
into interpretable Range-Doppler maps and Micro-Doppler spectrograms.
These are the primary inputs for the AI layer.

Functions:
1. dechirp_signal: Mixes Rx with Tx* to extract beat frequencies.
2. compute_range_doppler_map: 2D-FFT process.
3. compute_spectrogram: STFT process.
4. extract_features: Statistical feature vector generation.

Author: Principal Photonic Radar Scientist
"""

import numpy as np
from scipy.signal import stft, get_window
from typing import Tuple, Dict

def dechirp_signal(rx_signal: np.ndarray, tx_reference: np.ndarray) -> np.ndarray:
    """
    Performs de-chirping (mixing) of the received signal.
    
    Operation: S_IF = Rx * conj(Tx)
    
    Ideally, Tx reference should be the "analytic" signal. 
    If input is real, we might need Hilbert transform first, 
    but for this simulation, we assume complex baseband inputs if possible,
    or we handle mixing efficiently.
    
    If inputs are real: Mixing -> LPF -> Analytic (Hilbert) is standard.
    Here we assume inputs are complex baseband from the photonic simulation.
    """
    # Simple element-wise mixing
    # Returns the IF (Intermediate Frequency) signal
    return rx_signal * np.conj(tx_reference)

def get_adaptive_window(n_samples: int, method: str = 'hann', **kwargs) -> np.ndarray:
    """
    Generates a radar window function adapted to the signal processing requirements.
    
    Methods:
    - 'hann': Balanced (default).
    - 'cheb': Dolph-Chebyshev (prescribed sidelobe level).
    - 'taylor': Taylor (low sidelobes, broader mainlobe).
    - 'hamming': Traditional.
    """
    from scipy.signal.windows import chebwin, taylor
    
    if method == 'cheb':
        at = kwargs.get('at', 60) # Sidelobe attenuation in dB
        return chebwin(n_samples, at=at)
    elif method == 'taylor':
        nbar = kwargs.get('nbar', 4)
        sll = kwargs.get('sll', 35) # Sidelobe level
        return taylor(n_samples, nbar=nbar, sll=sll)
    else:
        # Fallback to scipy get_window for standard types
        return get_window(method, n_samples)

def compute_range_doppler_map(
    if_signal: np.ndarray, 
    n_chirps: int, 
    samples_per_chirp: int,
    n_fft_range: int = 128,
    n_fft_doppler: int = 128,
    window_method: str = 'hann'
) -> np.ndarray:
    """
    Computes Adaptive Range-Doppler Map using 2D FFT.
    
    Args:
        if_signal: 1D De-chirped signal array.
        n_chirps: Number of pulses (Slow-time dimension).
        samples_per_chirp: Samples per pulse (Fast-time dimension).
        window_method: Window type ('hann', 'cheb', 'taylor').
    
    Returns:
        rd_map: 2D Real-valued Magnitude Tensor (n_fft_doppler x n_fft_range).
                Doppler on axis 0, Range on axis 1.
    """
    # 1. Reshape to Matrix (Slow-Time x Fast-Time)
    total_samples = n_chirps * samples_per_chirp
    if len(if_signal) < total_samples:
        padded = np.zeros(total_samples, dtype=if_signal.dtype)
        padded[:len(if_signal)] = if_signal
        matrix = padded.reshape(n_chirps, samples_per_chirp)
    else:
        matrix = if_signal[:total_samples].reshape(n_chirps, samples_per_chirp)
    
    # 2. Adaptive Windowing
    # Range Windowing (Fast Time)
    win_range = get_adaptive_window(samples_per_chirp, method=window_method, at=80) 
    # Doppler Windowing (Slow Time)
    win_doppler = get_adaptive_window(n_chirps, method=window_method, at=60)
    
    matrix = matrix * win_range[np.newaxis, :]
    
    # 3. Fast-Time FFT (Range FFT) -> Axis 1
    range_profile = np.fft.fft(matrix, n=n_fft_range, axis=1)
    
    # Apply Doppler Window
    range_profile = range_profile * win_doppler[:, np.newaxis]
    
    # 4. Slow-Time FFT (Doppler FFT) -> Axis 0
    rd_complex = np.fft.fft(range_profile, n=n_fft_doppler, axis=0)
    
    # 5. Shift and Magnitude
    rd_shifted = np.fft.fftshift(rd_complex, axes=0)
    rd_mag = np.abs(rd_shifted)
    
    # Log scale
    rd_log = 20 * np.log10(rd_mag + 1e-9)
    
    return rd_log

def compute_spectrogram(
    signal: np.ndarray, 
    fs: float, 
    nperseg: int = 256, 
    noverlap: int = 128
) -> np.ndarray:
    """
    Computes Micro-Doppler Spectrogram using STFT.
    
    Returns:
        spectrogram: 2D Real-valued Magnitude Tensor (Frequency x Time).
    """
    f, t, Zxx = stft(signal, fs=fs, window='hann', nperseg=nperseg, noverlap=noverlap)
    
    # Magnitude
    mag = np.abs(Zxx)
    
    # Log scale
    spec_log = 20 * np.log10(mag + 1e-9)
    
    return spec_log

def extract_features(signal: np.ndarray) -> Dict[str, float]:
    """
    Extracts statistical features from the signal for lightweight analysis.
    Useful for sanity checks or fusion.
    """
    mag = np.abs(signal)
    phase = np.angle(signal)
    
    features = {
        "mean_power_db": float(10 * np.log10(np.mean(mag**2) + 1e-12)),
        "peak_power_db": float(10 * np.log10(np.max(mag**2) + 1e-12)),
        "phase_var": float(np.var(phase)),
        "kurtosis": float(np.mean((mag - np.mean(mag))**4) / (np.var(mag)**2 + 1e-12))
    }
    return features

if __name__ == "__main__":
    # Smoke Test
    # Simulate a chirp
    N = 4096
    t = np.linspace(0, 1, N)
    tx = np.exp(1j * 2 * np.pi * (100 * t + 500 * t**2)) # Chirp
    rx = np.roll(tx, 10) * 0.5 # Delayed + Attenuated
    
    # Dechirp
    if_sig = dechirp_signal(rx, tx)
    
    # RD Map
    rd = compute_range_doppler_map(if_sig, n_chirps=64, samples_per_chirp=64)
    
    # Spec
    spec = compute_spectrogram(if_sig, fs=4096)
    
    feat = extract_features(if_sig)
    
    print(f"RD Map shape: {rd.shape}, Mean dB: {np.mean(rd):.2f}")
    print(f"Spectrogram shape: {spec.shape}")
    print(f"Features: {feat}")
