"""
Radar Signal Processing: Core Transformation Module
===================================================

This module implements the fundamental Digital Signal Processing (DSP) algorithms 
required to extract target information from de-chirped photonic signals. 
It focuses on two-dimensional spectral estimation to separate targets in 
Range and Doppler dimensions.

Algorithmic Background:
-----------------------
1. De-chirping (Mixing):
   The received LFM signal (Rx) is mixed with a coherent copy of the transmitted 
   reference (Tx). In Frequency-Modulated Continuous-Wave (FMCW) radar, this 
   down-conversion results in a set of 'Beat' frequencies, where each frequency 
   is directly proportional to the target's range delay (tau).
   f_beat = (Sweep_Bandwidth / T_chirp) * tau

2. 2D Spectral Analysis (Range-Doppler):
   - Fast-Time FFT (Range): Processes samples within a single chirp. Identifies 
     the beat frequencies corresponding to target ranges.
   - Slow-Time FFT (Doppler): Processes the phase shift across multiple coherent 
     chirps. Identifies the Doppler shift (f_d) caused by target relative velocity.

3. Spectrogram (Micro-Doppler):
   Short-Time Fourier Transform (STFT) allows for time-resolved frequency analysis, 
   capturing non-stationary features like rotating drone blades or human gait.

Author: Lead DSP Research Engineer
"""

import numpy as np
from scipy.signal import stft, get_window
from typing import Tuple, Dict

def dechirp_signal(received_signal: np.ndarray, reference_signal: np.ndarray) -> np.ndarray:
    """
    Performs signal down-conversion (De-chirping) by mixing Rx with Tx*.
    
    This process collapses the wideband LFM ramp into a complex sinusoid (beat note) 
    at a frequency proportional to the target distance.
    """
    # complex baseband mixing: S_beat(t) = S_rx(t) * conj(S_tx(t))
    return received_signal * np.conj(reference_signal)

def get_specialized_window(n_samples: int, method: str = 'hann', **kwargs) -> np.ndarray:
    """
    Synthesizes radar window functions with optimized sidelobe suppression.
    
    Supported Methods:
    - 'hann': Baseline for spectral analysis (31dB sidelobe suppression).
    - 'cheb' (Dolph-Chebyshev): Minimum mainlobe width for a fixed sidelobe level.
    - 'taylor': Standard for radar imaging; prevents mainlobe broadening while suppressing tails.
    """
    from scipy.signal.windows import chebwin, taylor
    
    if method == 'cheb' or method == 'dolph':
        at_db = kwargs.get('at', 80) # Sidelobe attenuation target
        return chebwin(n_samples, at=at_db)
    elif method == 'taylor':
        nbar = kwargs.get('nbar', 4)
        sll_db = kwargs.get('sll', 40) # Sidelobe level target
        return taylor(n_samples, nbar=nbar, sll=sll_db)
    else:
        return get_window(method, n_samples)

def compute_range_doppler_map(
    beat_signal_complex: np.ndarray, 
    num_pulses: int, 
    samples_per_pulse: int,
    n_fft_range: int = 128,
    n_fft_doppler: int = 128,
    window_method: str = 'hann'
) -> np.ndarray:
    """
    Computes the 2D Range-Doppler Response via Sequential Fast-Time and Slow-Time FFTs.
    
    Args:
        beat_signal_complex: 1D concatenated complex beat signal.
        num_pulses: Count of coherent processing intervals (CPI).
        samples_per_pulse: Samples collected per chirp.
        n_fft_range: Range FFT size (Zero-padding for interpolation).
        n_fft_doppler: Doppler FFT size.
        window_method: Apodization method to minimize spectral leakage.
    
    Returns:
        range_doppler_intensity_db: 2D array (Doppler x Range) in decibels.
    """
    # 1. Structure raw data into Matrix (CPI-Pulse Index x Fast-Time Sample Index)
    expected_samples = num_pulses * samples_per_pulse
    if len(beat_signal_complex) < expected_samples:
        padded_signal = np.pad(beat_signal_complex, (0, expected_samples - len(beat_signal_complex)))
        data_matrix = padded_signal.reshape(num_pulses, samples_per_pulse)
    else:
        data_matrix = beat_signal_complex[:expected_samples].reshape(num_pulses, samples_per_pulse)
    
    # 2. Apodization (Weighting to suppress range/Doppler sidelobes)
    range_window = get_specialized_window(samples_per_pulse, method=window_method, at=80) 
    doppler_window = get_specialized_window(num_pulses, method=window_method, at=60)
    
    # Apply Fast-Time window
    data_matrix = data_matrix * range_window[np.newaxis, :]
    
    # 3. Fast-Time FFT (Range Processing) -> Transformed across Axis 1
    # Note: 1/N scaling preserves the average power level
    range_profiles = np.fft.fft(data_matrix, n=n_fft_range, axis=1) / samples_per_pulse
    
    # 4. Slow-Time FFT (Doppler Processing) -> Transformed across Axis 0
    # Apply Doppler window before transition
    range_profiles *= doppler_window[:, np.newaxis]
    rd_complex = np.fft.fft(range_profiles, n=n_fft_doppler, axis=0) / num_pulses
    
    # 5. Zero-Frequency Centering and Magnitude Extraction
    # Doppler is centered at DC (0 velocity), shifted to the middle of the array
    rd_centered = np.fft.fftshift(rd_complex, axes=0)
    rd_magnitude = np.abs(rd_centered)
    
    # Convert to Power Scale (dB) for visualization and dynamic range compression
    # Clipping floor at -180dB to avoid log(0) singularities
    range_doppler_intensity_db = 20 * np.log10(rd_magnitude + 1e-9)
    
    return range_doppler_intensity_db

def compute_spectrogram(
    signal: np.ndarray, 
    sampling_rate_hz: float, 
    nperseg: int = 256, 
    noverlap: int = 128
) -> np.ndarray:
    """
    Generates a Time-Frequency Spectrogram (Micro-Doppler Analysis).
    
    Returns the log-magnitude intensity plot.
    """
    f, t, Zxx = stft(signal, fs=sampling_rate_hz, window='hann', nperseg=nperseg, noverlap=noverlap)
    
    intensity_mag = np.abs(Zxx)
    intensity_db = 20 * np.log10(intensity_mag + 1e-9)
    
    return intensity_db

def extract_statistical_features(signal_complex: np.ndarray) -> Dict[str, float]:
    """
    Extracts higher-order statistical features from the pulse stream.
    
    Features:
    - Mean/Peak Power: SNR proxy.
    - Phase Variance: Measure of signal coherence/scintillation.
    - Kurtosis: Identifies impulsiveness (Target vs. Gaussian noise).
    """
    envelope = np.abs(signal_complex)
    instantaneous_phase = np.angle(signal_complex)
    
    features = {
        "mean_power_db": float(10 * np.log10(np.mean(envelope**2) + 1e-12)),
        "peak_power_db": float(10 * np.log10(np.max(envelope**2) + 1e-12)),
        "phase_stability_var": float(np.var(instantaneous_phase)),
        "envelope_kurtosis": float(np.mean((envelope - np.mean(envelope))**4) / (np.var(envelope)**2 + 1e-12))
    }
    return features

if __name__ == "__main__":
    # Internal Unit Test (Synthetic Chirp)
    N_total = 4096
    t_axis = np.linspace(0, 1, N_total)
    tx_ref = np.exp(1j * 2 * np.pi * (100 * t_axis + 500 * t_axis**2)) 
    rx_echo = np.roll(tx_ref, 10) * 0.5 
    
    beat_sig = dechirp_signal(rx_echo, tx_ref)
    
    rd_res = compute_range_doppler_map(beat_sig, num_pulses=64, samples_per_pulse=64)
    print(f"[DSP-TEST] Range-Doppler synthesized: {rd_res.shape}")
    print(f"[DSP-TEST] Feature vector: {extract_statistical_features(beat_sig)}")
