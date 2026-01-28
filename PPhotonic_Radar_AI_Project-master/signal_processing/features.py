"""
Radar Feature Extraction and Characterization
==============================================

This module provides high-level descriptors of the radar return signal, 
facilitating target classification and system diagnostics.

Supported Features:
-------------------
1. Range-Doppler Intensity: 2D spectral energy distribution.
2. Micro-Doppler Spectrograms: Time-varying spectral signatures.
3. Signal Coherence Metrics: Phase stability and statistical moments.
4. Photonic-Radar Parameters: Propagation-specific metrics (TTD, Chirp Slope).

Author: Senior Radar DSP Engineer
"""

import numpy as np
from typing import Dict, Tuple, Optional
from signal_processing.transforms import compute_range_doppler_map, compute_spectrogram

def extract_range_doppler_intensity(beat_signal_complex: np.ndarray, 
                                    num_pulses: int = 64, 
                                    samples_per_pulse: int = 64, 
                                    fft_size: int = 128) -> np.ndarray:
    """
    Synthesizes the 2D Range-Doppler response.
    Delegates to the core transformation layer.
    """
    return compute_range_doppler_map(
        beat_signal_complex, 
        num_pulses=num_pulses, 
        samples_per_pulse=samples_per_pulse,
        n_fft_range=fft_size, 
        n_fft_doppler=fft_size
    )

def extract_micro_doppler_spectrogram(beat_signal_complex: np.ndarray, 
                                      sampling_rate_hz: float = 4096, 
                                      window_length: int = 256) -> np.ndarray:
    """
    Synthesizes the Time-Frequency intensity map for micro-motion analysis.
    """
    return compute_spectrogram(beat_signal_complex, sampling_rate_hz=sampling_rate_hz, nperseg=window_length)

def compute_coherence_and_phase_metrics(signal_complex: np.ndarray) -> Dict[str, float]:
    """
    Analyzes the phase stability (coherence) of the complex signal stream.
    
    Metrics:
    - Coherence: Measured as the magnitude of the mean phasor.
    - Phase Variance: Degree of phase fluctuation.
    """
    instantaneous_phase = np.angle(signal_complex)
    
    # Coherence (Complex Mean Magnitude)
    # Range [0, 1] where 1 is perfectly coherent
    signal_coherence = np.abs(np.mean(np.exp(1j * instantaneous_phase)))
    
    return {
        "mean_phase_rad": float(np.mean(instantaneous_phase)),
        "phase_variance_rad2": float(np.var(instantaneous_phase)),
        "complex_coherence_index": float(signal_coherence)
    }

def estimate_photonic_derived_parameters(signal_complex: np.ndarray, 
                                         sweep_bandwidth_hz: float = 1e9, 
                                         chirp_duration_s: float = 10e-6) -> Dict:
    """
    Estimates derived radar parameters specific to photonic LFM signals.
    """
    # Simulated True-Time-Delay (TTD) calibration vector
    # Represents the phase alignment of an 8-element photonic phased array
    calibration_ttd_vector = np.exp(-1j * 2 * np.pi * 0.5 * np.arange(8))
    
    # Noise/Clutter Power Estimation (Tail-sample approximations)
    local_noise_floor = np.var(signal_complex[-100:]) if len(signal_complex) > 100 else 0.01
    
    chirp_slope_hz_s = sweep_bandwidth_hz / chirp_duration_s
    
    return {
        "instantaneous_bandwidth_hz": sweep_bandwidth_hz,
        "chirp_slope_hz_s": chirp_slope_hz_s,
        "chirp_duration_s": chirp_duration_s,
        "ttd_magnitude_array": np.abs(calibration_ttd_vector).tolist(),
        "estimated_noise_floor_power": float(local_noise_floor)
    }

def aggregate_tactical_features(beat_signal_complex: np.ndarray, 
                                sampling_rate_hz: float = 4096, 
                                rd_map_override: Optional[np.ndarray] = None) -> Tuple:
    """
    Aggregates all extracted features into a unified tactical descriptor.
    """
    if rd_map_override is None:
        range_doppler_map = extract_range_doppler_intensity(beat_signal_complex)
    else:
        range_doppler_map = rd_map_override
        
    spectrogram = extract_micro_doppler_spectrogram(beat_signal_complex, sampling_rate_hz=sampling_rate_hz)
    coherence_metrics = compute_coherence_and_phase_metrics(beat_signal_complex)
    photonic_stats = estimate_photonic_derived_parameters(beat_signal_complex)
    
    # Serialization for ML/AI compatibility
    feature_vector_linear = np.array([
        coherence_metrics['mean_phase_rad'],
        coherence_metrics['phase_variance_rad2'],
        coherence_metrics['complex_coherence_index'],
        photonic_stats['instantaneous_bandwidth_hz'],
        photonic_stats['chirp_slope_hz_s'],
        photonic_stats['chirp_duration_s'],
        photonic_stats['estimated_noise_floor_power']
    ], dtype=np.float32)
    
    return range_doppler_map, spectrogram, feature_vector_linear, photonic_stats
