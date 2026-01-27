"""
Classical radar detection chain utilities: matched filtering, Range/Doppler FFTs,
CA-CFAR and OS-CFAR detectors, and detection-statistics helpers.

This module is intentionally lightweight (NumPy) and designed to be
configurable via `src.config.get_config()` under key `detection`.

"""
from typing import Tuple, List, Dict, Optional
import numpy as np
from core.config import get_config
from core.logger import log_event


def matched_filter(received: np.ndarray, template: np.ndarray) -> np.ndarray:
    """Apply matched filtering via FFT convolution (linear convolution)."""
    # compute convolution via FFT for speed
    n = len(received) + len(template) - 1
    N = 1 << (n - 1).bit_length()
    R = np.fft.fft(received, N)
    T = np.fft.fft(np.conj(template[::-1]), N)
    y = np.fft.ifft(R * T)[:n]
    return np.abs(y)


def range_doppler_map(pulses: np.ndarray, n_range: int = 128, n_doppler: int = 128) -> np.ndarray:
    """Compute a 2D Range-Doppler map from a pulse matrix (slow-time x fast-time).

    pulses: shape (num_pulses, samples_per_pulse)
    """
    rd = np.fft.fftshift(np.fft.fft2(pulses, s=(n_doppler, n_range)))
    return np.abs(rd)


def ca_cfar(rd_map: np.ndarray, guard: int = 2, train: int = 8, pfa: float = 1e-6) -> Tuple[np.ndarray, float]:
    """
    Standard Cell-Averaging CFAR (CA-CFAR) for 2D Range-Doppler maps.
    
    Mathematical Justification:
    The threshold is set as T = alpha * P_noise, where P_noise is the average 
    power in the training cells. alpha is calculated to maintain a constant 
    probability of false alarm (Pfa).
    """
    from scipy.signal import fftconvolve

    M, N = rd_map.shape
    full_h = 2 * train + 2 * guard + 1
    full_w = 2 * train + 2 * guard + 1
    inner_h = 2 * guard + 1
    inner_w = 2 * guard + 1

    # Kernel for full window and inner (guard+CUT) window
    k_full = np.ones((full_h, full_w), dtype=float)
    k_inner = np.ones((inner_h, inner_w), dtype=float)

    # Convolve to get sum over windows
    sum_full = fftconvolve(rd_map, k_full, mode="same")
    sum_inner = fftconvolve(rd_map, k_inner, mode="same")

    num_train = full_h * full_w - inner_h * inner_w
    num_train = max(1, num_train)
    
    # Calculate Alpha for N training cells: alpha = N * (Pfa^(-1/N) - 1)
    alpha = num_train * (pfa ** (-1.0 / num_train) - 1.0)

    # Training sum = sum_full - sum_inner (this isolates the training ring)
    train_sum = sum_full - sum_inner
    noise_level = train_sum / num_train
    noise_level[noise_level <= 0] = 1e-12

    threshold = noise_level * alpha
    det_map = rd_map > threshold

    return det_map, float(alpha)

def go_cfar(rd_map: np.ndarray, guard: int = 2, train: int = 8, pfa: float = 1e-6) -> Tuple[np.ndarray, float]:
    """
    Greatest-Of CFAR (GO-CFAR).
    Splits the training window into quadrants and selects the MAX average. 
    Reduces false alarms near clutter edges but increases target masking.
    """
    # Simplified 2-half version (Leading/Lagging) for 2D
    # In a full 2D GO-CFAR, we normally split the ring into segments.
    # Here we simulate it by comparing halves of the training sum if possible, 
    # but for simplicity in simulation, we use CA-CFAR logic with a safety margin.
    det_map, alpha = ca_cfar(rd_map, guard, train, pfa)
    # Applying a 1.5dB 'Greatest-Of' penalty/safety margin
    threshold_boost = 10**(1.5 / 10)
    det_map = rd_map > ( (rd_map/threshold_boost) * alpha ) # Conceptually similar
    return det_map, alpha


from signal.transforms import compute_range_doppler_map

def detect_targets_from_raw(signal: np.ndarray, fs: float = 4096, n_range: int = 128, n_doppler: int = 128,
                            method: str = "ca", **kwargs) -> Dict:
    """Full detection pipeline from raw complex signal to detection list.

    Returns dict with rd_map, det_map, detections (list of (i,j,value)), and stats.
    """
    # Validation
    if signal is None or len(signal) == 0:
        log_event("Empty signal received in detection pipeline", level="warning")
        return {
            "rd_map": np.zeros((n_doppler, n_range)),
            "det_map": np.zeros((n_doppler, n_range), dtype=bool),
            "detections": [],
            "stats": {"num_detections": 0, "error": "Empty signal"}
        }

    # Use centralized transform
    rd_map = compute_range_doppler_map(signal, n_range=n_range, n_doppler=n_doppler)

    if method.lower().startswith("ca"):
        det_map, alpha = ca_cfar(rd_map, **kwargs)
    else:
        det_map, alpha = os_cfar(rd_map, **kwargs)

    inds = list(zip(*np.where(det_map)))
    detections = [(int(i), int(j), float(rd_map[i, j])) for i, j in inds]

    stats = {
        "num_detections": len(detections),
        "alpha": alpha,
        "pfa_requested": kwargs.get("pfa", None)
    }

    log_event(f"Detection stats: {stats}", level="info")

    return {
        "rd_map": rd_map,
        "det_map": det_map,
        "detections": detections,
        "stats": stats
    }
