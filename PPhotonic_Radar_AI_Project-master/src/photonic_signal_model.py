"""
Lightweight photonic radar signal models.

Features:
- Laser phase noise (simple Wiener process)
- Optical heterodyne RF generation (beat of two lasers)
- Time-delay beamforming jitter (small random time offsets)
- Coherence loss modeled from temperature drift

Configurable via `src.config.get_config()` under key `photonic_model` in `config.yaml`.

Deterministic behavior via `seed` parameter.

Equations (inline):

- Laser phase noise (Wiener):
  $\\phi(t+\\Delta t) = \\phi(t) + \\sqrt{2\\pi \\Delta f}\\,\\sqrt{\\Delta t}\\,N(0,1)$
  where $\\Delta f$ is laser linewidth in Hz. This integrates white frequency noise to phase.

- Heterodyne RF (optical beat):
  $s_{RF}(t) = \\mathrm{Re}\\{E_1(t) e^{j\\omega_1 t} + E_2(t) e^{j\\omega_2 t}\\}^2 \\approx A\\cos((\\omega_1-\\omega_2)t + \\phi_1(t)-\\phi_2(t))$

- Time-delay jitter: apply small random delays $\\tau_i \\sim \\mathcal{N}(0,\\sigma_{\\tau}^2)$ per channel to emulate jitter.

- Coherence loss from temperature drift: $\\mathrm{coherence}(t) = \\exp(-|\\Delta T(t)|/T_{\\mathrm{coh}})$

The implementation favors simplicity and speed (NumPy-only); suitable for simulation pipelines.

Example:
>>> from src.photonic_signal_model import generate_photonic_rf
>>> t, rf = generate_photonic_rf(duration=0.01, fs=20000, seed=42)

"""
from typing import Optional, Dict, Tuple
import numpy as np

from .config import get_config


def _get_model_cfg() -> Dict:
    cfg = get_config()
    return cfg.get("photonic_model", {
        "laser_linewidth_hz": 1e3,
        "optical_freq_hz": 193.414e12,
        "local_osc_freq_offset_hz": 1e9,
        "num_channels": 1,
        "beamjitter_std_ns": 1.0,  # nanoseconds
        "temp_drift_rate_C_per_s": 0.01,
        "temp_coherence_scale_C": 5.0,
        "amplitude": 1.0,
    })


def _wiener_phase_noise(linewidth_hz: float, dt: float, n: int, rng: np.random.Generator) -> np.ndarray:
    """Generate phase noise samples using a discrete Wiener process.

    Phase increment variance per step: Var(dphi) = 2*pi*linewidth * dt
    dphi = sqrt(2*pi*linewidth*dt) * N(0,1)
    """
    sigma = np.sqrt(2.0 * np.pi * linewidth_hz * dt)
    dphi = rng.normal(scale=sigma, size=n)
    phi = np.cumsum(dphi)
    return phi


def _temperature_drift(duration: float, fs: float, rate_C_per_s: float) -> np.ndarray:
    """Simple linear temperature drift plus small low-frequency fluctuations."""
    t = np.arange(int(np.ceil(duration * fs))) / fs
    base = rate_C_per_s * t
    # gentle low-freq sinusoidal variation to emulate diurnal / slow changes
    fluct = 0.1 * np.sin(2 * np.pi * 0.01 * t)
    return base + fluct


def generate_photonic_rf(duration: float,
                         fs: float = 1e4,
                         num_channels: int = 1,
                         seed: Optional[int] = None,
                         cfg_override: Optional[Dict] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a lightweight photonic RF signal produced by optical heterodyne and photodetection.

    Returns (t, rf_signals) where `rf_signals` shape is (num_channels, samples)

    Parameters:
    - duration: seconds
    - fs: sampling frequency (Hz)
    - num_channels: number of beamforming channels (applies independent jitter)
    - seed: optional int for deterministic RNG
    - cfg_override: optional dict to override YAML config values locally
    """
    model_cfg = _get_model_cfg()
    if cfg_override:
        model_cfg.update(cfg_override)

    rng = np.random.default_rng(seed)

    N = int(np.ceil(duration * fs))
    t = np.arange(N) / fs
    dt = 1.0 / fs

    linewidth = float(model_cfg.get("laser_linewidth_hz", 1e3))
    opt_freq = float(model_cfg.get("optical_freq_hz", 193.414e12))
    lo_offset = float(model_cfg.get("local_osc_freq_offset_hz", 1e9))
    amp = float(model_cfg.get("amplitude", 1.0))
    beamjitter_ns = float(model_cfg.get("beamjitter_std_ns", 1.0))
    temp_rate = float(model_cfg.get("temp_drift_rate_C_per_s", 0.01))
    temp_scale = float(model_cfg.get("temp_coherence_scale_C", 5.0))

    # Laser phases (two lasers: signal + LO) using Wiener phase noise
    phi_sig = _wiener_phase_noise(linewidth, dt, N, rng)
    phi_lo = _wiener_phase_noise(linewidth, dt, N, rng)

    # Optical fields (complex envelope): E(t) = A * exp(j*(omega*t + phi(t)))
    # We keep omega large but work with difference frequency for RF after photodetection.
    # RF beat: cos((omega_sig - omega_lo)*t + phi_sig - phi_lo)
    delta_omega = 2.0 * np.pi * lo_offset
    rf_phase = delta_omega * t + (phi_sig - phi_lo)

    base_rf = amp * np.cos(rf_phase)

    # Temperature-driven coherence loss: multiplicative factor in amplitude
    temp = _temperature_drift(duration, fs, temp_rate)
    coherence = np.exp(-np.abs(temp) / max(1e-9, temp_scale))

    # Apply coherence envelope
    rf_coherent = base_rf * coherence

    # Beamforming jitter: support both 'fft' and 'sinc' fractional delay methods
    signals = np.zeros((num_channels, N), dtype=float)
    method = model_cfg.get("fractional_delay_method", "sinc").lower()
    max_ns = float(model_cfg.get("fractional_delay_max_ns", 5.0))

    def _fractional_delay_sinc(x: np.ndarray, tau: float, fs: float, kernel_len: int = 129) -> np.ndarray:
        """Apply a windowed-sinc fractional delay filter.

        tau: delay in seconds (can be fractional). Positive tau delays the signal.
        kernel_len: odd length of the sinc kernel (tradeoff fidelity vs cost)
        """
        # convert to fractional samples
        d = tau * fs
        M = int((kernel_len - 1) // 2)
        n = np.arange(-M, M + 1)
        # sinc kernel using numpy.sinc which takes argument x -> sinc(pi*x)/(pi*x) normalized
        h = np.sinc(n - d)
        # apply a Hann window for smoother roll-off
        w = np.hanning(len(h))
        h *= w
        # normalize to preserve DC gain
        h /= np.sum(h)
        # linear convolution (use 'full' then center-crop to length of x to avoid
        # numpy behavior where 'same' may return kernel-length when kernel>signal)
        y_full = np.convolve(x, h, mode='full')
        L = len(y_full)
        start = (L - len(x)) // 2
        y = y_full[start:start + len(x)]
        return y

    # Precompute FFT components only if needed
    if method == "fft":
        S = np.fft.rfft(rf_coherent)
        freqs = np.fft.rfftfreq(N, d=dt)  # Hz
        omega = 2.0 * np.pi * freqs

    for ch in range(num_channels):
        # jitter in seconds (clamped)
        tau_ns = rng.normal(loc=0.0, scale=beamjitter_ns)
        tau_ns = np.clip(tau_ns, -max_ns, max_ns)
        tau = float(tau_ns) * 1e-9

        if method == "fft":
            phase_ramp = np.exp(-1j * omega * tau)
            S_shifted = S * phase_ramp
            shifted = np.fft.irfft(S_shifted, n=N)
        else:
            # default to windowed-sinc fractional delay in time-domain
            shifted = _fractional_delay_sinc(rf_coherent, tau, fs, kernel_len=129)

        # small additional per-sample multiplicative jitter to emulate amplitude jitter
        per_sample_jitter = rng.normal(loc=1.0, scale=0.001, size=N)
        signals[ch] = shifted * per_sample_jitter

    return t, signals


def example_usage():
    """Simple smoke test when run as a script."""
    t, s = generate_photonic_rf(duration=0.01, fs=20000, num_channels=4, seed=1234)
    print("Generated", s.shape, "samples")


if __name__ == "__main__":
    example_usage()
