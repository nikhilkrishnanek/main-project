"""
Radar Noise, Clutter, and Electronic Interference Modeling
==========================================================

This module provides high-fidelity stochastic models for simulating the 
complex electromagnetic environments encountered by radar systems.

Physical Models:
----------------
1. AWGN (Additive White Gaussian Noise): Represents the collective thermal noise 
   from the antenna and receiver low-noise amplifier (LNA).
2. Clutter: Unwanted echoes from the environment.
   - Weibull: Effective for modeling sea clutter at low grazing angles.
   - K-Distribution: Standard for high-resolution sea and land clutter with 
     heavy-tailed (non-Rayleigh) characteristics.
3. Electronic Interference (EA): Models for intentional jamming, including 
   Spot Jamming (narrowband) and Barrage Jamming (swept/wideband).

Author: Senior Radar Systems Engineer (EM Modeling Division)
"""

import numpy as np
from typing import Optional


def inject_thermal_awgn(signal_complex: np.ndarray, snr_db: float) -> np.ndarray:
    """
    Simulates Additive White Gaussian Noise (AWGN) based on a target Signal-to-Noise Ratio.
    """
    signal_power = np.mean(np.abs(signal_complex)**2)
    noise_power_linear = signal_power / (10**(snr_db / 10.0))
    
    # Complex circular symmetric Gaussian noise
    standard_deviation = np.sqrt(noise_power_linear / 2.0)
    noise_samples = standard_deviation * (np.random.randn(*signal_complex.shape) + 1j * np.random.randn(*signal_complex.shape))
    
    return signal_complex + noise_samples


def generate_stochastic_clutter(num_samples: int, 
                                distribution: str = 'weibull', 
                                tactical_scenario: str = 'custom', 
                                **kwargs) -> np.ndarray:
    """
    Synthesizes non-Gaussian radar clutter samples for specific tactical environments.
    """
    if tactical_scenario == 'urban':
        # Dense urban terrain: High texture K-distribution
        return generate_stochastic_clutter(num_samples, distribution='k', shape=1.2, scale=2.0)
    elif tactical_scenario == 'maritime':
        # Sea clutter: Spiky Weibull distribution
        return generate_stochastic_clutter(num_samples, distribution='weibull', shape=0.7, scale=1.5)
    elif tactical_scenario == 'arid':
        # Desert/Flat terrain: Approximates Rayleigh (Gaussian IQ)
        return generate_stochastic_clutter(num_samples, distribution='gaussian')
        
    if distribution == 'weibull':
        # Weibull distribution for magnitude; uniform distribution for phase
        scale_param = kwargs.get('scale', 1.0)
        shape_param = kwargs.get('shape', 1.5)
        clutter_magnitude = scale_param * np.random.weibull(shape_param, num_samples)
    elif distribution == 'k':
        # K-distribution: Product of Gamma (texture) and Gaussian (speckle)
        nu_shape = kwargs.get('shape', 2.0)
        mu_scale = kwargs.get('scale', 1.0)
        texture_component = np.random.gamma(nu_shape, mu_scale, num_samples)
        speckle_component = (np.random.randn(num_samples) + 1j * np.random.randn(num_samples)) / np.sqrt(2.0)
        return np.sqrt(texture_component) * speckle_component
    else:
        # Default to Rayleigh/Gaussian clutter
        return (np.random.randn(num_samples) + 1j * np.random.randn(num_samples)) / np.sqrt(2.0)
    
    random_phases = np.random.uniform(0, 2 * np.pi, num_samples)
    return clutter_magnitude * np.exp(1j * random_phases)


def apply_spatial_clutter_overlay(range_doppler_intensity_db: np.ndarray, 
                                 clutter_profile: str = 'ground') -> np.ndarray:
    """
    Simulates environmental clutter artifacts on a processed Range-Doppler map.
    """
    num_doppler_bins, num_range_bins = range_doppler_intensity_db.shape
    clutter_mask = np.zeros_like(range_doppler_intensity_db)
    
    if clutter_profile == 'ground':
        # Stationary ground clutter: Concentrated at low ranges and near-zero Doppler
        center_bin = num_doppler_bins // 2
        clutter_rows = slice(center_row - 4, center_row + 5)
        clutter_cols = slice(0, 25)
        
        # Aggregate Weibull clutter components
        clutter_samples = generate_stochastic_clutter(9 * 25, distribution='weibull', shape=1.3, scale=4.0)
        clutter_mask[clutter_rows, clutter_cols] = np.abs(clutter_samples.reshape(9, 25))
        
    elif clutter_profile == 'meteorological':
        # Rain/Weather clutter: Broad spectral spread across mid-range
        weather_width = num_range_bins // 5
        clutter_samples = generate_stochastic_clutter(num_doppler_bins * weather_width, distribution='k', shape=2.5)
        clutter_mask[:, num_range_bins // 2 : num_range_bins // 2 + weather_width] = \
            np.abs(clutter_samples.reshape(num_doppler_bins, weather_width)) * 2.5
        
    return range_doppler_intensity_db + clutter_mask


def inject_electronic_interference(signal_complex: np.ndarray, 
                                   jamming_type: str = 'narrowband_spot', 
                                   **kwargs) -> np.ndarray:
    """
    Simulates hostile electronic attack (EA) signals.
    """
    sampling_rate_hz = kwargs.get('sampling_rate_hz', 1.0)
    num_samples = len(signal_complex)
    time_axis = np.arange(num_samples) / sampling_rate_hz
    
    if jamming_type == 'narrowband_spot':
        # Constant frequency tone jammer
        freq_hz = kwargs.get('freq_hz', 0.1 * sampling_rate_hz)
        amplitude = kwargs.get('amplitude', 1.0)
        jamming_signal = amplitude * np.exp(1j * 2 * np.pi * freq_hz * time_axis)
    elif jamming_type == 'barrage_sweep':
        # Swept-frequency/Chirp jammer
        f_start = kwargs.get('f_start_hz', 0.0)
        f_stop = kwargs.get('f_stop_hz', sampling_rate_hz / 2.0)
        jamming_signal = np.exp(1j * 2 * np.pi * (f_start * time_axis + (f_stop - f_start) / (2 * time_axis[-1]) * time_axis**2))
    else:
        jamming_signal = np.zeros_like(signal_complex)
        
    return signal_complex + jamming_signal
