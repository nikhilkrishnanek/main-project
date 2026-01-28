"""
Photonic Radar Physics & Component Modeling
===========================================

This module provides high-fidelity, yet computationally efficient, analytical models 
for key microwave photonic components. It serves as the physical backbone for the 
radar simulation pipeline.

Supported Physical Phenomena:
-----------------------------
1. Laser Phase Noise: Modeled as a discrete-time Wiener process, representing 
   phase fluctuations due to spontaneous emission within the laser cavity.
2. Optical Heterodyne Mixing: Simulations of RF beat-note generation through 
   square-law photodetection of dual laser sources.
3. Thermal Coherence Decay: Models the degradation of signal phase stability 
   due to ambient temperature-induced refractive index drifts.
4. TTD Jitter (True Time Delay): Fractional delay modeling for photonic beamforming, 
   accounting for timing jitter in fiber networks.

Author: Senior Radar Systems Engineer (MWP Division)
"""

from typing import Optional, Dict, Tuple
import numpy as np

from core.config import get_config
from photonic_models.comb import generate_flat_comb
from photonic_models.transmission import simulate_wdm_channel
from photonic_models.beamforming import calculate_squint_error


def _fetch_physics_config() -> Dict:
    """Retrieves physics-level hyperparameters from the global configuration."""
    cfg = get_config()
    return cfg.get("photonic_model", {
        "laser_linewidth_hz": 1e3,
        "optical_center_freq_hz": 193.414e12, # 1550nm C-Band reference
        "local_oscillator_offset_hz": 1e9,
        "beamforming_channels": 1,
        "delay_jitter_std_ns": 1.0, 
        "thermal_drift_rate_K_per_s": 0.01,
        "coherence_temperature_scale_K": 5.0,
        "amplitude_v": 1.0,
    })


def _generate_wiener_phase_noise(laser_linewidth_hz: float, 
                                sampling_period_s: float, 
                                num_samples: int, 
                                rng: np.random.Generator) -> np.ndarray:
    """
    Synthesizes laser phase noise samples using a discrete Wiener process.
    
    The phase variance accumulates linearly with time: Var[phi(t)] = 2 * pi * linewidth * t
    """
    phase_step_std = np.sqrt(2.0 * np.pi * laser_linewidth_hz * sampling_period_s)
    phase_increments = rng.normal(scale=phase_step_std, size=num_samples)
    return np.cumsum(phase_increments)


def _simulate_thermal_environment(duration_s: float, 
                                  sampling_rate_hz: float, 
                                  drift_rate_K_per_s: float) -> np.ndarray:
    """Models slow temperature variations affecting photonic component stability."""
    time_axis = np.arange(int(np.ceil(duration_s * sampling_rate_hz))) / sampling_rate_hz
    # Linear drift + diurnal sinusoidal fluctuation
    thermal_profile = (drift_rate_K_per_s * time_axis) + 0.1 * np.sin(2 * np.pi * 0.01 * time_axis)
    return thermal_profile


def generate_heterodyne_rf_signal(duration_s: float,
                                  sampling_rate_hz: float = 1e4,
                                  num_channels: int = 1,
                                  seed: Optional[int] = None,
                                  config_override: Optional[Dict] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates a photonic RF signal via simulated optical heterodyne detection.

    Args:
        duration_s: Observation time in seconds.
        sampling_rate_hz: Sample frequency in Hz.
        num_channels: Count of independent spatial/beamforming channels.
        seed: Determinism seed for RNG.
        config_override: Local override for physics parameters.

    Returns:
        (time_axis, rf_waveforms): Time vector and multi-channel RF voltage matrix.
    """
    physics_cfg = _fetch_physics_config()
    if config_override:
        physics_cfg.update(config_override)

    rng = np.random.default_rng(seed)

    num_samples = int(np.ceil(duration_s * sampling_rate_hz))
    time_axis = np.arange(num_samples) / sampling_rate_hz
    sampling_period_s = 1.0 / sampling_rate_hz

    # Parameter Extraction
    linewidth = float(physics_cfg.get("laser_linewidth_hz", 1e3))
    lo_freq_offset = float(physics_cfg.get("local_oscillator_offset_hz", 1e9))
    amplitude = float(physics_cfg.get("amplitude_v", 1.0))
    jitter_ns = float(physics_cfg.get("delay_jitter_std_ns", 1.0))
    
    # FMCW Sweep Parameters
    sweep_bandwidth = float(physics_cfg.get("fmcw_bandwidth_hz", 0.0))
    chirp_period = float(physics_cfg.get("fmcw_chirp_period_s", 1e-3))

    # 1. Phase Noise Synthesis (Independent Signal/LO Laser Sources)
    phi_signal_laser = _generate_wiener_phase_noise(linewidth, sampling_period_s, num_samples, rng)
    phi_local_osc_laser = _generate_wiener_phase_noise(linewidth, sampling_period_s, num_samples, rng)

    # 2. FMCW Modulation (Phase Interpolation)
    if sweep_bandwidth > 0 and chirp_period > 0:
        time_modulated = np.mod(time_axis, chirp_period)
        sweep_slope = sweep_bandwidth / chirp_period
        chirp_phase = 2.0 * np.pi * (0.5 * sweep_slope * time_modulated**2)
    else:
        chirp_phase = 0.0

    # 3. RF Beat-note Construction
    # Phase(t) = 2*pi*f_offset*t + Phi_Chirp(t) + [Phi_Sig(t) - Phi_LO(t)]
    fundamental_phase = (2.0 * np.pi * lo_freq_offset * time_axis) + chirp_phase + (phi_signal_laser - phi_local_osc_laser)
    raw_rf_voltage = amplitude * np.cos(fundamental_phase)

    # 4. Thermal Coherence Attenuation
    # Models SNR degradation as components drift out of thermal equilibrium
    thermal_profile = _simulate_thermal_environment(duration_s, sampling_rate_hz, 
                                                   float(physics_cfg.get("thermal_drift_rate_K_per_s", 0.01)))
    coherence_scale = float(physics_cfg.get("coherence_temperature_scale_K", 5.0))
    coherence_loss = np.exp(-np.abs(thermal_profile) / max(1e-9, coherence_scale))
    coherent_rf = raw_rf_voltage * coherence_loss

    # 5. Photonic True Time Delay (TTD) Jitter Modeling
    channel_waveforms = np.zeros((num_channels, num_samples), dtype=float)
    delay_method = physics_cfg.get("fractional_delay_method", "sinc").lower()
    max_jitter_ns = float(physics_cfg.get("fractional_delay_max_ns", 5.0))

    def _apply_fractional_delay_sinc(signal_in: np.ndarray, 
                                     delay_s: float, 
                                     fs_hz: float, 
                                     window_len: int = 129) -> np.ndarray:
        """Applies high-fidelity time-shifting using a windowed-sinc interpolator."""
        delay_samples = delay_s * fs_hz
        m_half = (window_len - 1) // 2
        offsets = np.arange(-m_half, m_half + 1)
        # Sinc kernel calculation
        sinc_kernel = np.sinc(offsets - delay_samples) * np.hanning(window_len)
        sinc_kernel /= np.sum(sinc_kernel) # Energy normalization
        # Convolution and latency compensation
        y_conv = np.convolve(signal_in, sinc_kernel, mode='full')
        return y_conv[m_half : m_half + len(signal_in)]

    # Pre-compute FFT if using frequency-domain delay
    if delay_method == "fft":
        spectrum = np.fft.rfft(coherent_rf)
        angular_freqs = 2.0 * np.pi * np.fft.rfftfreq(num_samples, d=sampling_period_s)

    for ch in range(num_channels):
        # Sample random jitter per channel
        tau_jitter_ns = np.clip(rng.normal(0, jitter_ns), -max_jitter_ns, max_jitter_ns)
        tau_jitter_s = float(tau_jitter_ns) * 1e-9

        if delay_method == "fft":
            phase_shift = np.exp(-1j * angular_freqs * tau_jitter_s)
            channel_waveforms[ch] = np.fft.irfft(spectrum * phase_shift, n=num_samples)
        else:
            channel_waveforms[ch] = _apply_fractional_delay_sinc(coherent_rf, tau_jitter_s, sampling_rate_hz)

        # Micro-amplitude jitter (Residual Intensity Noise - RIN proxy)
        channel_waveforms[ch] *= rng.normal(1.0, 0.001, size=num_samples)

    return time_axis, channel_waveforms


def run_physics_demo():
    """Simple diagnostic script for physics engine validation."""
    time, sigs = generate_heterodyne_rf_signal(duration_s=0.01, sampling_rate_hz=20000, num_channels=2, seed=42)
    print(f"[PHYSICS-CORE] Generated {sigs.shape} waveforms. RMS: {np.sqrt(np.mean(sigs**2)):.4e}")


if __name__ == "__main__":
    run_physics_demo()
