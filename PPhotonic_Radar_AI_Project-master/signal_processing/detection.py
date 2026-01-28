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


def ca_cfar(rd_map: np.ndarray, guard: int = 2, train: int = 8, pfa: float = 1e-4, min_pwr: float = 1e-3) -> Tuple[np.ndarray, float]:
    """
    Standard Cell-Averaging CFAR (CA-CFAR) for 2D Range-Doppler maps.
    Optimized for Square-Law detectors (Power).
    Includes a min_pwr threshold to filter out extreme noise-floor triggers.
    """
    from scipy.signal import fftconvolve

    # Ensure rd_map is in linear power
    # Square window dimensions
    full_side = 2 * train + 2 * guard + 1
    inner_side = 2 * guard + 1

    # Kernel for full window and inner (guard+CUT) window
    k_full = np.ones((full_side, full_side), dtype=float)
    k_inner = np.ones((inner_side, inner_side), dtype=float)

    # Convolve to get sum over windows
    sum_full = fftconvolve(rd_map, k_full, mode="same")
    sum_inner = fftconvolve(rd_map, k_inner, mode="same")

    num_train = (full_side**2) - (inner_side**2)
    num_train = max(1, num_train)
    
    # ALPHA CALCULATION (Square-law detector)
    alpha = num_train * (pfa ** (-1.0 / num_train) - 1.0)

    # Training sum = total window sum - guard/CUT sum
    train_sum = sum_full - sum_inner
    
    # Mean noise power estimation
    noise_level = train_sum / num_train
    noise_level[noise_level <= 0] = 1e-12

    threshold = noise_level * alpha
    
    # Combined detection: CFAR threshold and absolute power floor
    det_map = (rd_map > threshold) & (rd_map > min_pwr)

    return det_map, float(alpha)

def os_cfar(rd_map: np.ndarray, guard: int = 2, train: int = 8, pfa: float = 1e-6, k_rank: Optional[int] = None) -> Tuple[np.ndarray, float]:
    """
    Ordered-Statistics CFAR (OS-CFAR).
    More robust in multi-target environments (prevents target masking).
    """
    from scipy.ndimage import generic_filter
    
    m, n = rd_map.shape
    full_side = 2 * train + 2 * guard + 1
    inner_side = 2 * guard + 1
    num_train = (full_side**2) - (inner_side**2)
    
    # Default rank k = 3/4 of training cells is a common defensive standard
    if k_rank is None:
        k_rank = int(0.75 * num_train)
    
    # Define footprint mask (exclude guard cells and CUT)
    footprint = np.ones((full_side, full_side), dtype=bool)
    start_inner = train
    end_inner = train + inner_side
    footprint[start_inner:end_inner, start_inner:end_inner] = False
    
    def os_threshold(buffer):
        # Buffer contains only training cells due to footprint
        sorted_cells = np.sort(buffer)
        return sorted_cells[k_rank - 1]

    # Sliding window OS estimation (Note: this is computationally expensive compared to convolution)
    noise_est = generic_filter(rd_map, os_threshold, footprint=footprint, mode='constant', cval=0.0)
    
    # OS-CFAR Alpha calculation (Approximate for Rayleigh noise)
    # T = alpha * X_k
    # Approximation for high k and low Pfa:
    alpha = (-np.log(pfa))**(1.0/k_rank) # Simplified approximation
    # For strict defense, we use lookup tables or more complex analytical forms.
    # Here we use a tuned constant that provides reasonable performance.
    alpha = 10**(3.0 / 10) # 3dB margin for the k-th statistic in this simulation
    
    threshold = noise_est * alpha
    det_map = rd_map > threshold
    
    return det_map, float(alpha)


def cluster_detections(det_map: np.ndarray, rd_map: np.ndarray) -> List[Tuple[int, int]]:
    """
    Groups connected detection pixels and picks the local maximum for each cluster.
    Prevents one target from generating multiple tracks.
    """
    from scipy.ndimage import label
    
    if not np.any(det_map):
        return []
        
    labeled_array, num_features = label(det_map)
    centroids = []
    
    for i in range(1, num_features + 1):
        # Find indices of this cluster
        coords = np.argwhere(labeled_array == i)
        
        # Pick the coordinate with the maximum power in rd_map
        cluster_powers = [rd_map[r, c] for r, c in coords]
        max_idx = np.argmax(cluster_powers)
        centroids.append(tuple(coords[max_idx]))
        
    return centroids


def detect_targets_from_raw(signal: np.ndarray, fs: float = 4096, n_range: int = 128, n_doppler: int = 128,
                            method: str = "ca", cluster: bool = True, **kwargs) -> Dict:
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
    from signal_processing.transforms import compute_range_doppler_map
    rd_map = compute_range_doppler_map(signal, n_range=n_range, n_doppler=n_doppler)

    if method.lower().startswith("ca"):
        det_map, alpha = ca_cfar(rd_map, **kwargs)
    else:
        det_map, alpha = os_cfar(rd_map, **kwargs)

    if cluster:
        inds = cluster_detections(det_map, rd_map)
    else:
        inds = list(zip(*np.where(det_map)))
        
    detections = [(int(i), int(j), float(rd_map[i, j])) for i, j in inds]

    stats = {
        "num_detections": len(detections),
        "alpha": alpha,
        "pfa_requested": kwargs.get("pfa", None),
        "clustered": cluster
    }

    log_event(f"Detection stats: {stats}", level="info")

    return {
        "rd_map": rd_map,
        "det_map": det_map,
        "detections": detections,
        "stats": stats
    }
