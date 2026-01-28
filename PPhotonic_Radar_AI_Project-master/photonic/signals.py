"""
Photonic Radar Signal Synthesis Module
======================================

This module implements the generation of wideband radar waveforms using Microwave Photonics (MWP)
principles. It specifically models the synthesis of Linear Frequency Modulated (LFM) signals
through optical heterodyning of laser tones.

Physics Fundamentals:
--------------------
1. Optical Heterodyning: 
   Mixing two optical fields (E_sig and E_lo) on a high-speed photodetector produces an 
   RF beat note. The detector behaves as a square-law device:
   I_photo(t) ∝ |E_sig(t) + E_lo(t)|^2 
   The resulting AC photocurrent contains the frequency difference (beat note).

2. Laser Phase Noise:
   The coherence of the generated RF signal is limited by the linewidth (Δν) of the source lasers.
   Phase noise is modeled as a Wiener process (Random Walk), representing the instantaneous 
   frequency fluctuations of the laser.

3. FMCW Modulation:
   Linear frequency sweeping is applied in the optical domain, then translated to the RF domain
   via heterodyne mixing. This ermöglicht (enables) extremely high sweep bandwidths (>10 GHz) 
   typical of photonic systems.

Author: Lead Senior Photonic Radar Engineer
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional

@dataclass
class PhotonicConfig:
    """Design parameters for the Microwave Photonic RF Source."""
    sampling_rate_hz: float = 20e9        # ADC/DAC sampling rate (e.g., 20 GS/s)
    chirp_duration_s: float = 10e-6       # Coherent pulse width (T_p)
    start_frequency_hz: float = 8e9       # Initial carrier frequency (f_c)
    sweep_bandwidth_hz: float = 4e9       # Total LFM sweep bandwidth (B)
    laser_linewidth_hz: float = 100e3     # FWHM spectral width of the laser source
    optical_power_dbm: float = 10.0      # Average optical power per line (P_opt)
    photodetector_responsivity: float = 0.8 # Responsivity (R) in A/W
    number_of_comb_lines: int = 1         # Count of coherent optical carriers (OFC mode)
    comb_line_spacing_hz: float = 10e9    # Frequency separation in the optical comb

    # Adaptive Scaling Factors (Driven by Cognitive Loop)
    bandwidth_scaling_factor: float = 1.0
    transmit_power_scaling_factor: float = 1.0

def generate_laser_phase_noise(num_samples: int, 
                               sampling_rate_hz: float, 
                               linewidth_hz: float, 
                               rng: np.random.Generator) -> np.ndarray:
    """
    Synthesizes laser phase noise following a Wiener process.
    
    The phase increment Δφ between samples is normally distributed with:
    Variance(Δφ) = 2 * pi * linewidth * Δt
    
    This represents the integral of white frequency noise (random walk phase).
    """
    delta_t = 1.0 / sampling_rate_hz
    # Standard deviation of the phase step (Lorentzian lineshape model)
    sigma_phase_step = np.sqrt(2 * np.pi * linewidth_hz * delta_t)
    
    # Generate independent normally distributed increments
    phase_increments = rng.normal(0, sigma_phase_step, num_samples)
    
    # Cumulative integration to form the phase random walk
    return np.cumsum(phase_increments)

def generate_optical_comb(config: PhotonicConfig, 
                          time_vector: np.ndarray, 
                          rng: np.random.Generator) -> np.ndarray:
    """
    Simulates a Phase-Locked Optical Frequency Comb (OFC).
    
    All comb lines share a common phase noise source, representing the 
    internal coherence of a single-laser-driven comb synthesizer.
    """
    total_optical_field = np.zeros(len(time_vector), dtype=complex)
    power_watts = 10 ** ((config.optical_power_dbm - 30) / 10)
    amplitude = np.sqrt(power_watts)
    
    # Common Phase Noise (Maintains relative phase stability between lines)
    phi_common = generate_laser_phase_noise(len(time_vector), 
                                            config.sampling_rate_hz, 
                                            config.laser_linewidth_hz, 
                                            rng)
    
    for i in range(config.number_of_comb_lines):
        frequency_offset = i * config.comb_line_spacing_hz
        # Construct line: E_i = A * exp(j * (2*pi*f_i*t + phi_v))
        total_optical_field += amplitude * np.exp(1j * (2 * np.pi * frequency_offset * time_vector + phi_common))
        
    return total_optical_field

def generate_synthetic_photonic_signal(config: PhotonicConfig, 
                                      seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    End-to-end simulation of the MWP Radar Transmitter.
    
    Returns:
        time_vector (np.ndarray): Discrete time axis
        photocurrent_rf (np.ndarray): Generated RF signal (Photodetector output voltage)
    """
    rng = np.random.default_rng(seed)
    num_samples = int(config.chirp_duration_s * config.sampling_rate_hz)
    time_vector = np.arange(num_samples) / config.sampling_rate_hz
    
    # 1. Optical Infrastructure Setup
    power_watts = 10 ** ((config.optical_power_dbm - 30) / 10)
    
    # 2. Transmit Line Generation (Signal Path)
    if config.number_of_comb_lines > 1:
        # Multi-carrier transmission (OFC)
        optical_field_tx = generate_optical_comb(config, time_vector, rng)
        # Apply LFM modulation to the comb carriers
        chirp_slope = (config.sweep_bandwidth_hz * config.bandwidth_scaling_factor) / config.chirp_duration_s
        phase_modulation = 2 * np.pi * (config.start_frequency_hz * time_vector + 0.5 * chirp_slope * time_vector**2)
        optical_field_tx *= np.exp(1j * phase_modulation)
    else:
        # Single carrier transmission
        phase_noise_tx = generate_laser_phase_noise(num_samples, 
                                                   config.sampling_rate_hz, 
                                                   config.laser_linewidth_hz, 
                                                   rng)
        chirp_slope = (config.sweep_bandwidth_hz * config.bandwidth_scaling_factor) / config.chirp_duration_s
        phase_modulation = 2 * np.pi * (config.start_frequency_hz * time_vector + 0.5 * chirp_slope * time_vector**2)
        
        # Weighted by Transmit Power Scaling (from Cognitive Engine)
        amplitude_tx = np.sqrt(power_watts * config.transmit_power_scaling_factor)
        optical_field_tx = amplitude_tx * np.exp(1j * (phase_modulation + phase_noise_tx))
        
    # 3. Local Oscillator (LO) Generation (Reference Path)
    # Modeled as a stable, single-tone laser source for heterodyning
    phase_noise_lo = generate_laser_phase_noise(num_samples, 
                                               config.sampling_rate_hz, 
                                               config.laser_linewidth_hz, 
                                               rng)
    optical_field_lo = np.sqrt(power_watts) * np.exp(1j * phase_noise_lo)
    
    # 4. Square-Law Photodetection (RF Conversion)
    # The AC photocurrent is proportional to the real part of (E_tx * conj(E_lo))
    # I_ac = 2 * R * Re{ E_tx * E_lo* }
    photocurrent_rf = 2 * config.photodetector_responsivity * np.real(optical_field_tx * np.conj(optical_field_lo))
    
    return time_vector, photocurrent_rf

if __name__ == "__main__":
    # Diagnostic Self-Test
    test_config = PhotonicConfig(chirp_duration_s=1e-6, sweep_bandwidth_hz=1e9)
    time, signal = generate_synthetic_photonic_signal(test_config, seed=42)
    print(f"[PHOTONIC-RADAR] Signal Synthesis Success.")
    print(f"Samples: {len(signal)} | Std Dev: {np.std(signal):.4e} Volts")
